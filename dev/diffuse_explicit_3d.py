import taichi as ti
import pyvista as pv
ti.init(arch = ti.gpu, device_memory_fraction=0.9)

# advection-diffusion equation
# https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation

# problem setting
nx = 50
ny = 50
nz = 50
scatter = 8
resx = nx * scatter
resy = ny * scatter
resz = nz * scatter

# physical parameters
# time-integration related
h = 1e-3    # time-step size
substep = 1 # number of substeps
dx = 1      # finite difference step size (in space)
dy = 1
dz = 1
nd = 3

# heat-source related
t_max = 300 # ti.max temperature (in Celsius)
t_min = 0   # ti.min temperature 
heat_center = (nx//2, ny//2, nz//2) 
heat_radius = 1
k = 50.0 # scale for rate of heat diffusion

# visualization
pixels = ti.Vector.field(3, ti.f32, shape = (resx, resy, resz))

# diffuse matrix
n2 = nx * ny * nz

# temperature now and temperature next_time
temp = ti.field(ti.f32, shape = n2,) # unrolled to 1-d array
temp_1 = ti.field(ti.f32, shape = n2,) # unrolled to 1-d array
temp_1_sm = ti.field(ti.f32, shape = n2,) # unrolled to 1-d array
grad_temp = ti.Vector.field(nd, ti.f32, shape = n2,)
update = ti.field(ti.f32, shape = n2,)
velocity = ti.Vector.field(nd, ti.f32, shape = n2,) # from fluid
diffusivity = ti.field(ti.f32, shape = n2,) # from fluid sim (equal in all directions)
v_temp = ti.Vector.field(nd, ti.f32, shape = n2,)

@ti.func
def ind(i, j, k): return i*ny*nz + j*nz + k


@ti.kernel
def gradient_temp():
    for i, j, k in ti.ndrange(nx,ny,nz):
        x1 = i+1
        x2 = i+2
        x_1 = i-1
        x_2 = i-2
        x_1 = ti.max(0, x_1)
        x_2 = ti.max(0, x_2)
        x1 = ti.min(nx-1, x1)
        x2 = ti.min(nx-1, x2)
        y1 = j+1
        y2 = j+2
        y_1 = j-1
        y_2 = j-2
        y_1 = ti.max(0, y_1)
        y_2 = ti.max(0, y_2)
        y1 = ti.min(ny-1, y1)
        y2 = ti.min(ny-1, y2)
        z1 = k+1
        z2 = k+2
        z_1 = k-1
        z_2 = k-2
        z_1 = ti.max(0, z_1)
        z_2 = ti.max(0, z_2)
        z1 = ti.min(nz-1, z1)
        z2 = ti.min(nz-1, z2)
        
        xd = (2/3)*(temp[ind(x1,j,k)] - temp[ind(x_1,j,k)]) + (1/12)*(temp[ind(x2,j,k)] - temp[ind(x_2,j,k)])
        yd = (2/3)*(temp[ind(i,y1,k)] - temp[ind(i,y_1,k)]) + (1/12)*(temp[ind(i,y2,k)] - temp[ind(i,y_2,k)])
        zd = (2/3)*(temp[ind(i,j,z1)] - temp[ind(i,j,z_1)]) + (1/12)*(temp[ind(i,j,z2)] - temp[ind(i,j,z_2)])
        grad_temp[ind(i,j,k)] = ti.Vector([xd/dx, yd/dy, zd/dz])

@ti.kernel
def smoothfilter(A: ti.template(), B: ti.template()):
    for i,j,k in ti.ndrange(nx,ny,nz):
        B[ind(i,j,k)] = (A[ind(i-1,j,k)] + 2*A[ind(i,j,k)] + A[ind(i+1,j,k)])/4
    for i,j,k in ti.ndrange(nx,ny,nz):
        A[ind(i,j,k)] = (B[ind(i,j-1,k)] + 2*B[ind(i,j,k)] + B[ind(i,j+1,k)])/4
    for i,j,k in ti.ndrange(nx,ny,nz):
        B[ind(i,j,k)] = (A[ind(i,j,k-1)] + 2*A[ind(i,j,k)] + A[ind(i,j,k+1)])/4
        
        

@ti.kernel
def divergence(A:ti.template(), d: ti.template(), scale:float):
    for i, j, k in ti.ndrange(nx,ny,nz):
        x1 = i+1
        x2 = i+2
        x_1 = i-1
        x_2 = i-2
        x_1 = ti.max(0, x_1)
        x_2 = ti.max(0, x_2)
        x1 = ti.min(nx-1, x1)
        x2 = ti.min(nx-1, x2)
        y1 = j+1
        y2 = j+2
        y_1 = j-1
        y_2 = j-2
        y_1 = ti.max(0, y_1)
        y_2 = ti.max(0, y_2)
        y1 = ti.min(ny-1, y1)
        y2 = ti.min(ny-1, y2)
        z1 = k+1
        z2 = k+2
        z_1 = k-1
        z_2 = k-2
        z_1 = ti.max(0, z_1)
        z_2 = ti.max(0, z_2)
        z1 = ti.min(nz-1, z1)
        z2 = ti.min(nz-1, z2)
        
        xd = (2/3)*(A[ind(x1,j,k)][0] - A[ind(x_1,j,k)][0]) + (1/12)*(A[ind(x2,j,k)][0] - A[ind(x_2,j,k)][0])
        yd = (2/3)*(A[ind(i,y1,k)][1] - A[ind(i,y_1,k)][1]) + (1/12)*(A[ind(i,y2,k)][1] - A[ind(i,y_2,k)][1])
        zd = (2/3)*(A[ind(i,j,z1)][2] - A[ind(i,j,z_1)][2]) + (1/12)*(A[ind(i,j,z2)][2] - A[ind(i,j,z_2)][2])
        d[ind(i,j,k)] += (xd/dx + yd/dy + zd/dz) * scale

@ti.kernel
def mult(A:ti.template(), B: ti.template(), C: ti.template()):
    for i, j, k in ti.ndrange(nx, ny, nz):
        C[ind(i,j,k)] = A[ind(i,j,k)] * B[ind(i,j,k)]

@ti.kernel
def inplace_mult(A:ti.template(), B: ti.template()):
    for i, j, k in ti.ndrange(nx, ny, nz):
        A[ind(i,j,k)] = A[ind(i,j,k)] * B[ind(i,j,k)]

@ti.kernel
def scale(A: ti.template(), scale: float):
    for i, j, k in ti.ndrange(nx, ny, nz):
        A[ind(i,j,k)] *= scale

@ti.kernel
def add(A: ti.template(), B: ti.template(), C: ti.template()):
    for i, j, k in ti.ndrange(nx, ny, nz):
        C[ind(i,j,k)] = A[ind(i,j,k)] + B[ind(i,j,k)]

 
@ti.kernel
def init():
    for i, j, k in ti.ndrange(nx,ny,nz):
        if (float(i)-heat_center[0])**2 + (float(j)-heat_center[1])**2 + (float(j)-heat_center[2])**2 <= heat_radius**2:
            temp[ind(i, j, k)] = t_max # source
            temp_1[ind(i, j, k)] = t_max # source
        else:
            temp[ind(i, j, k)] = t_min
            temp_1[ind(i, j, k)] = t_min

@ti.kernel
def update_source():
    for i, j, k in ti.ndrange(nx,ny,nz):
        if (float(i)-heat_center[0])**2 + (float(j)-heat_center[1])**2 + (float(j)-heat_center[2])**2 <= heat_radius**2:
            temp_1_sm[ind(i, j, k)] = t_max


def diffuse(dt: ti.f32):
    gradient_temp()
    inplace_mult(grad_temp,diffusivity)
    update.fill(0.0)
    divergence(grad_temp, update, k) # diffusion
    mult(velocity,temp,v_temp)
    divergence(v_temp, update, -1.0) # advection
    scale(update, dt)
    add(temp, update, temp_1)
    smoothfilter(temp_1,temp_1_sm) # important to avoid numerical instability & checkerboard artifacts from square grid
    # if we can implement an better finite difference that uses a 3x3 stencil, we can remove the smoothing step
    
    # d_temp_dt = k * del( D  @ del(temp)) - del( v @ temp) + sources

def update_source_and_commit():
    update_source()
    temp.copy_from(temp_1_sm)

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
def temperature_to_color(t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32):
    for i, j, k in ti.ndrange(nx,ny,nz):
        for p,q,r in ti.ndrange(scatter, scatter, scatter):
            color[i*scatter+p,j*scatter+q,k*scatter+r] = get_color(t[ind(i,j,k)], tmin, tmax)


init()
velocity.fill(ti.Vector([0.0, 0.0, 0.0]))
diffusivity.fill(1.0)
pl = pv.Plotter()
pl.open_gif(f"images/output_3d.gif")
for i in range(20000):
    
    for _ in range(substep):
        diffuse(h/substep)
        update_source_and_commit()

    if i % 1000 == 0:
        print(f"Step {i}")
        # temperature_to_color(temp_1, pixels, t_min, t_max)
        tmp = temp_1.to_numpy().reshape((nx,ny,nz))
        # print(tmp.shape)
        pl.add_volume(tmp, cmap="jet", clim=[t_min, t_max])
        pl.write_frame()
pl.close()
        
        
        