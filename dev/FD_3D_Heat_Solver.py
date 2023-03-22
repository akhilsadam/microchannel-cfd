import taichi as ti
import pyvista as pv
import numpy as np
# ti.init(arch = ti.gpu, device_memory_fraction=0.9)

@ti.data_oriented
class FD_3D_Heat_Solver:
    # advection-diffusion equation
    # https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation

    def __init__(self,
                 nx,ny,nz,
                 h = 1e-3, substep = 1, dx = 1, dy = 1, dz = 1,
                 k = 1.0,
                 velocity_ref = None,
                 save_images = False, filename='heat'):
        # problem setting
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.it = 0
        self.plot_freq = 2000

        # physical parameters
        # time-integration related
        self.h = h    # time-step size
        self.substep = substep # number of substeps
        self.dx = dx      # finite difference step size (in space)
        self.dy = dy
        self.dz = dz
        self.nd = 3
    
        # heat-source related
        self.t_max = 343.15 # ti.max temperature (in Celsius)
        self.t_min = 293.15   # ti.min temperature 
        self.heat_center = (self.nx//2, self.ny//2, self.nz//2)
        self.heat_radius = 6
        self.k = k # scale for rate of heat diffusion

        self.max_update = 8000

        # diffuse matrix
        n2 = self.nx * self.ny * self.nz

        # temperature now and temperature next_time
        self.temp = ti.field(ti.f32, shape = n2,) # unrolled to 1-d array
        self.temp_1 = ti.field(ti.f32, shape = n2,) # unrolled to 1-d array
        self.temp_1_sm = ti.field(ti.f32, shape = n2,) # unrolled to 1-d array
        self.grad_temp = ti.Vector.field(self.nd, ti.f32, shape = n2,)
        self.update = ti.field(ti.f32, shape = n2,)
        self.velocity = velocity_ref # from fluid
        self.diffusivity = ti.field(ti.f32, shape = n2,) # from fluid sim (equal in all directions)
        self.v_temp = ti.Vector.field(self.nd, ti.f32, shape = n2,)


        # STARTUP
        self.init()
        
        if save_images:
            self.pl = pv.Plotter()
            self.pl.open_gif(f"images/{filename}.gif")

    @ti.func
    def ind(self, i, j, k): return i*self.ny*self.nz + j*self.nz + k


    @ti.kernel
    def gradient_temp(self):
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx,self.ny,self.nz):
            x1 = i+1
            x2 = i+2
            x_1 = i-1
            x_2 = i-2
            x_1 = ti.max(0, x_1)
            x_2 = ti.max(0, x_2)
            x1 = ti.min(self.nx-1, x1)
            x2 = ti.min(self.nx-1, x2)
            y1 = j+1
            y2 = j+2
            y_1 = j-1
            y_2 = j-2
            y_1 = ti.max(0, y_1)
            y_2 = ti.max(0, y_2)
            y1 = ti.min(self.ny-1, y1)
            y2 = ti.min(self.ny-1, y2)
            z1 = k+1
            z2 = k+2
            z_1 = k-1
            z_2 = k-2
            z_1 = ti.max(0, z_1)
            z_2 = ti.max(0, z_2)
            z1 = ti.min(self.nz-1, z1)
            z2 = ti.min(self.nz-1, z2)
            xd = 0.0
            yd = 0.0
            zd = 0.0
            
            if i > 1 and i < self.nx-2:
                xd = (2/3)*(self.temp[self.ind(x1,j,k)] - self.temp[self.ind(x_1,j,k)]) + (1/12)*(self.temp[self.ind(x2,j,k)] - self.temp[self.ind(x_2,j,k)])
            elif i <= 1:
                # forward difference
                xd = self.temp[self.ind(x1,j,k)] - self.temp[self.ind(i,j,k)]
            else:
                # backward difference
                xd = self.temp[self.ind(i,j,k)] - self.temp[self.ind(x_1,j,k)]
                
            if j > 1 and j < self.ny-2:
                yd = (2/3)*(self.temp[self.ind(i,y1,k)] - self.temp[self.ind(i,y_1,k)]) + (1/12)*(self.temp[self.ind(i,y2,k)] - self.temp[self.ind(i,y_2,k)])
            elif j <= 1:
                # forward difference
                yd = self.temp[self.ind(i,y1,k)] - self.temp[self.ind(i,j,k)]
            else:
                yd = self.temp[self.ind(i,j,k)] - self.temp[self.ind(i,y_1,k)]
            
            if k > 1 and k < self.nz-2:
                zd = (2/3)*(self.temp[self.ind(i,j,z1)] - self.temp[self.ind(i,j,z_1)]) + (1/12)*(self.temp[self.ind(i,j,z2)] - self.temp[self.ind(i,j,z_2)])
            elif k <= 1:
                # forward difference
                zd = self.temp[self.ind(i,j,z1)] - self.temp[self.ind(i,j,k)]
            else:
                zd = self.temp[self.ind(i,j,k)] - self.temp[self.ind(i,j,z_1)]
            
            self.grad_temp[self.ind(i,j,k)] = ti.Vector([xd/self.dx, yd/self.dy, zd/self.dz])

    @ti.kernel
    def smoothfilter(self, A: ti.template(), B: ti.template()):
        ti.loop_config(block_dim=256)
        for i,j,k in ti.ndrange(self.nx,self.ny,self.nz):
            if i>0 and i<self.nx-1:
                B[self.ind(i,j,k)] = (A[self.ind(i-1,j,k)] + 2*A[self.ind(i,j,k)] + A[self.ind(i+1,j,k)])/4
            else:
                B[self.ind(i,j,k)] = A[self.ind(i,j,k)]
        ti.loop_config(block_dim=256)
        for i,j,k in ti.ndrange(self.nx,self.ny,self.nz):
            if j>0 and j<self.ny-1:
                A[self.ind(i,j,k)] = (B[self.ind(i,j-1,k)] + 2*B[self.ind(i,j,k)] + B[self.ind(i,j+1,k)])/4
            else:
                A[self.ind(i,j,k)] = B[self.ind(i,j,k)]
        ti.loop_config(block_dim=256)
        for i,j,k in ti.ndrange(self.nx,self.ny,self.nz):
            if k>0 and k<self.nz-1:
                B[self.ind(i,j,k)] = (A[self.ind(i,j,k-1)] + 2*A[self.ind(i,j,k)] + A[self.ind(i,j,k+1)])/4
            else:
                B[self.ind(i,j,k)] = A[self.ind(i,j,k)]
            
            

    @ti.kernel
    def divergence(self, A:ti.template(), d: ti.template(), scale:float):
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx,self.ny,self.nz):
            x1 = i+1
            x2 = i+2
            x_1 = i-1
            x_2 = i-2
            x_1 = ti.max(0, x_1)
            x_2 = ti.max(0, x_2)
            x1 = ti.min(self.nx-1, x1)
            x2 = ti.min(self.nx-1, x2)
            y1 = j+1
            y2 = j+2
            y_1 = j-1
            y_2 = j-2
            y_1 = ti.max(0, y_1)
            y_2 = ti.max(0, y_2)
            y1 = ti.min(self.ny-1, y1)
            y2 = ti.min(self.ny-1, y2)
            z1 = k+1
            z2 = k+2
            z_1 = k-1
            z_2 = k-2
            z_1 = ti.max(0, z_1)
            z_2 = ti.max(0, z_2)
            z1 = ti.min(self.nz-1, z1)
            z2 = ti.min(self.nz-1, z2)
            
            xd = 0.0
            yd = 0.0
            zd = 0.0
            
            if i > 1 and i < self.nx-2:
                xd = (2/3)*(A[self.ind(x1,j,k)][0] - A[self.ind(x_1,j,k)][0]) + (1/12)*(A[self.ind(x2,j,k)][0] - A[self.ind(x_2,j,k)][0])
            elif i <= 1:
                xd = (A[self.ind(x1,j,k)][0] - A[self.ind(i,j,k)][0])
            else:
                xd = (A[self.ind(i,j,k)][0] - A[self.ind(x_1,j,k)][0])        
            
            if j > 1 and j < self.ny-2:
                yd = (2/3)*(A[self.ind(i,y1,k)][1] - A[self.ind(i,y_1,k)][1]) + (1/12)*(A[self.ind(i,y2,k)][1] - A[self.ind(i,y_2,k)][1])
            elif j <= 1:
                # forward difference
                yd = (A[self.ind(i,y1,k)][1] - A[self.ind(i,j,k)][1])
            else:
                # backward difference
                yd = (A[self.ind(i,j,k)][1] - A[self.ind(i,y_1,k)][1])
            
            if k > 1 and k < self.nz-2:
                zd = (2/3)*(A[self.ind(i,j,z1)][2] - A[self.ind(i,j,z_1)][2]) + (1/12)*(A[self.ind(i,j,z2)][2] - A[self.ind(i,j,z_2)][2])
            elif k <= 1:
                zd = (A[self.ind(i,j,z1)][2] - A[self.ind(i,j,k)][2])
            else:
                zd = (A[self.ind(i,j,k)][2] - A[self.ind(i,j,z_1)][2])
        
            d[self.ind(i,j,k)] += (xd/self.dx + yd/self.dy + zd/self.dz) * scale

    @ti.kernel
    def clamp(self, A: ti.template(), clamp: float):
        mean = 0.0
        count = 0.0
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            val = ti.abs(A[self.ind(i,j,k)])
            if val > 0.0:
                count += 1.0
            mean += val
        mean /= count
        maxval = ti.max(mean*clamp,self.max_update)
        
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if maxval < ti.abs(A[self.ind(i,j,k)]):
                # print("WARNING: Detected Floating Point Overflow. Clamping...")
                
                xvg = 0.0
                if i>0 and i<self.nx-1:
                    xvg = (A[self.ind(i+1,j,k)] + A[self.ind(i-1,j,k)])/2
                elif i==0:
                    xvg = A[self.ind(i+1,j,k)]
                else:
                    xvg = A[self.ind(i-1,j,k)]
                    
                yvg = 0.0
                
                if j>0 and j<self.ny-1:
                    yvg = (A[self.ind(i,j+1,k)] + A[self.ind(i,j-1,k)])/2
                elif j==0:
                    yvg = A[self.ind(i,j+1,k)]
                else:
                    yvg = A[self.ind(i,j-1,k)]
                
                zvg = 0.0
                if k>0 and k<self.nz-1:
                    zvg = (A[self.ind(i,j,k+1)] + A[self.ind(i,j,k-1)])/2
                elif k==0:
                    zvg = A[self.ind(i,j,k+1)]
                else:
                    zvg = A[self.ind(i,j,k-1)]
                    
                avg = (xvg + yvg + zvg)/3
                    
                if ti.abs(avg) > maxval:
                    print("CRITICAL: Deleted region of value ", A[self.ind(i,j,k)], " at ", i, j, k)
                    avg = 0.0
                    
                A[self.ind(i,j,k)] = avg

    @ti.kernel
    def mult(self, A:ti.template(), B: ti.template(), C: ti.template()):
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            C[self.ind(i,j,k)] = A[i,j,k] * B[self.ind(i,j,k)]

    @ti.kernel
    def inplace_mult(self, A:ti.template(), B: ti.template()):
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            A[self.ind(i,j,k)] = A[self.ind(i,j,k)] * B[self.ind(i,j,k)]

    @ti.kernel
    def scale(self, A: ti.template(), scale: float):
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            A[self.ind(i,j,k)] *= scale

    @ti.kernel
    def add(self, A: ti.template(), B: ti.template(), C: ti.template()):
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            C[self.ind(i,j,k)] = A[self.ind(i,j,k)] + B[self.ind(i,j,k)]

    
    @ti.kernel
    def init(self):
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx,self.ny,self.nz):
            if (float(i)-self.heat_center[0])**2 + (float(j)-self.heat_center[1])**2 + (float(k)-self.heat_center[2])**2 <= self.heat_radius**2:
                self.temp[self.ind(i, j, k)] = self.t_max # source
                self.temp_1[self.ind(i, j, k)] = self.t_max # source
            else:
                self.temp[self.ind(i, j, k)] = self.t_min
                self.temp_1[self.ind(i, j, k)] = self.t_min

    @ti.kernel
    def update_source(self):
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx,self.ny,self.nz):
            if (float(i)-self.heat_center[0])**2 + (float(j)-self.heat_center[1])**2 + (float(k)-self.heat_center[2])**2 <= self.heat_radius**2:
                self.temp_1_sm[self.ind(i, j, k)] = self.t_max


    def diffuse(self, dt: ti.f32):
        self.gradient_temp()
        self.inplace_mult(self.grad_temp, self.diffusivity)
        self.update.fill(0.0)
        self.divergence(self.grad_temp, self.update, self.k) # diffusion
        self.mult(self.velocity, self.temp, self.v_temp)
        self.divergence(self.v_temp, self.update, -1.0) # advection
        # print(np.max(self.update.to_numpy()))       
        self.clamp(self.update, 2.0) # avoid numerical instability, but this is a hack
        
        self.scale(self.update, dt)
        
        self.add(self.temp, self.update, self.temp_1)
    
        self.smoothfilter(self.temp_1, self.temp_1_sm) # important to avoid numerical instability & checkerboard artifacts from square grid
        # if we can implement an better finite difference, we can remove the smoothing step
        
        # d_temp_dt = k * del( D  @ del(temp)) - del( v @ temp) + sources

    def update_source_and_commit(self):
        self.update_source()
        self.temp.copy_from(self.temp_1_sm)

    @staticmethod
    @ti.func
    def get_color(v, vmin, vmax):
        c = ti.Vector([1.0, 1.0, 1.0]) # white

        v = ti.max(v, vmin)
        v = ti.min(v, vmax)
        dv = vmax - vmin

        if v < (vmin + 0.25 * dv):
            c[0] = 0
            c[1] = 4 * (v-vmin) / dv
        elif v < (vmin + 0.5 * dv):
            c[0] = 0
            c[2] = 1 + 4 * (vmin + 0.25*dv -v) / dv
        elif v < (vmin + 0.75*dv):
            c[0] = 4 * (v - vmin -0.5 * dv) / dv
            c[2] = 0
        else:
            c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
            c[2] = 0

        return c

    @ti.kernel
    def temperature_to_color(self, t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32):
        ti.loop_config(block_dim=256)
        for i, j, k in ti.ndrange(self.nx,self.ny,self.nz):
            for p,q,r in ti.ndrange(self.scatter, self.scatter, self.scatter):
                color[i*self.scatter+p,j*self.scatter+q,k*self.scatter+r] = FD_3D_Heat_Solver.get_color(t[self.ind(i,j,k)], tmin, tmax)
    
    def update_diffusivity(self):
        self.diffusivity.fill(100.0)

    def cycle(self):        
        for _ in range(self.substep):
            self.update_diffusivity()
            
            self.diffuse(self.h/self.substep)
            self.update_source_and_commit()


        if self.pl is not None and self.it % self.plot_freq == 0:
            print(f"Step {self.it}")
            # temperature_to_color(temp_1, pixels, t_min, t_max)
            tmp = self.temp_1.to_numpy().reshape((self.nx,self.ny,self.nz))
            # print(tmp.shape)
            self.pl.add_volume(tmp, cmap="jet", clim=[self.t_min, self.t_max])
            self.pl.write_frame()

        self.it += 1
        
    def exit(self):
        if self.save_images:
            self.pl.close()
            
        
        