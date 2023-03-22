import taichi as ti
from taichi.linalg.sparse_matrix import SparseMatrixBuilder
ti.init(arch = ti.gpu)


# advection-diffusion equation
# https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation


# control
paused = True
save_images = False

# problem setting
nx = 65
ny = 129
scatter = 8
resx = nx * scatter
resy = ny * scatter

# physical parameters
# time-integration related
h = 1e-3    # time-step size
substep = 1 # number of substeps
dx = 1      # finite difference step size (in space)
dy = 1

# heat-source related
t_max = 300 # ti.max temperature (in Celsius)
t_min = 0   # ti.min temperature 
heat_center = (nx//2, ny//2) 
heat_radius = 1
k = 50.0 # scale for rate of heat diffusion

# visualization
pixels = ti.Vector.field(3, ti.f32, shape = (resx, resy))

# diffuse matrix
n2 = nx * ny
# D_builder = SparseMatrixBuilder(n2, n2, ti.max_num_triplets=n2*5)
I_builder = SparseMatrixBuilder(n2, n2, max_num_triplets=n2)

# temperature now and temperature next_time
temp = ti.field(ti.f32, shape = n2,) # unrolled to 1-d array
temp_1 = ti.field(ti.f32, shape = n2,) # unrolled to 1-d array
temp_1_sm = ti.field(ti.f32, shape = n2,) # unrolled to 1-d array
grad_temp = ti.Vector.field(2, ti.f32, shape = n2,)
update = ti.field(ti.f32, shape = n2,)
velocity = ti.Vector.field(2, ti.f32, shape = n2,) # from fluid
diffusivity = ti.field(ti.f32, shape = n2,) # from fluid sim (equal in all directions)
v_temp = ti.Vector.field(2, ti.f32, shape = n2,)

@ti.func
def ind(i, j): return i*ny+j


@ti.kernel
def gradient_temp():
    for i,j in ti.ndrange(nx, ny):
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
        xd = 0.0
        yd = 0.0
        
        if i > 1 and i < nx-2:
            xd = (2/3)*(temp[ind(x1,j)] - temp[ind(x_1,j)]) + (1/12)*(temp[ind(x2,j)] - temp[ind(x_2,j)])
        elif i <= 1:
            xd = (temp[ind(x1,j)] - temp[ind(i,j)])
        else:
            xd = (temp[ind(i,j)] - temp[ind(x_1,j)])
            
        if j > 1 and j < ny-2:
            yd = (2/3)*(temp[ind(i,y1)] - temp[ind(i,y_1)]) + (1/12)*(temp[ind(i,y2)] - temp[ind(i,y_2)])
        elif j <= 1:
            yd = (temp[ind(i,y1)] - temp[ind(i,j)])
        else:
            yd = (temp[ind(i,j)] - temp[ind(i,y_1)])
            
        grad_temp[ind(i,j)] = ti.Vector([xd/dx, yd/dy])

@ti.kernel
def smoothfilter(A: ti.template(), B: ti.template()):
    for i, j in ti.ndrange(nx,ny):
        # gaussian blur kernel of size 3x3, centered at (i,j)
        B[ind(i,j)] = (A[ind(i-1,j-1)] + 2*A[ind(i-1,j)] + A[ind(i-1,j+1)] \
                        + 2*A[ind(i,j-1)] + 4*A[ind(i,j)] + 2*A[ind(i,j+1)] \
                        + A[ind(i+1,j-1)] + 2*A[ind(i+1,j)] + A[ind(i+1,j+1)]) / 16
        

@ti.kernel
def divergence(A:ti.template(), d: ti.template(), scale:float):
    for i, j in ti.ndrange(nx, ny):
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
        xd = 0.0
        yd = 0.0
        
        if i > 1 and i < nx-2:
            xd = (2/3)*(A[ind(x1,j)][0] - A[ind(x_1,j)][0]) + (1/12)*(A[ind(x2,j)][0] - A[ind(x_2,j)][0])
        elif i <= 1:
            # forward difference
            xd = (A[ind(x1,j)][0] - A[ind(i,j)][0])
        else:
            # backward difference
            xd = (A[ind(i,j)][0] - A[ind(x_1,j)][0])
            
        if j > 1 and j < ny-2:
            yd = (2/3)*(A[ind(i,y1)][1] - A[ind(i,y_1)][1]) + (1/12)*(A[ind(i,y2)][1] - A[ind(i,y_2)][1])
        elif j <= 1:
            # forward difference
            yd = (A[ind(i,y1)][1] - A[ind(i,j)][1])
        else:
            # backward difference
            yd = (A[ind(i,j)][1] - A[ind(i,y_1)][1])
            
        d[ind(i,j)] += (xd/dx + yd/dy) * scale

@ti.kernel
def mult(A:ti.template(), B: ti.template(), C: ti.template()):
    for i, j in ti.ndrange(nx, ny):
        C[ind(i,j)] = A[ind(i,j)] * B[ind(i,j)]

@ti.kernel
def inplace_mult(A:ti.template(), B: ti.template()):
    for i, j in ti.ndrange(nx, ny):
        A[ind(i,j)] = A[ind(i,j)] * B[ind(i,j)]

@ti.kernel
def scale(A: ti.template(), scale: float):
    for i, j in ti.ndrange(nx, ny):
        A[ind(i,j)] *= scale

@ti.kernel
def add(A: ti.template(), B: ti.template(), C: ti.template()):
    for i, j in ti.ndrange(nx, ny):
        C[ind(i,j)] = A[ind(i,j)] + B[ind(i,j)]

# @ti.kernel
# def fillDiffusionMatrixBuilder(A: ti.sparse_matrix_builder()):
#     for i,j in ti.ndrange(nx, ny):
#         count = 0
#         if i-1 >= 0:
#             A[ind(i,j), ind(i-1,j)] += 1
#             count += 1
#         if i+1 < nx:
#             A[ind(i,j), ind(i+1,j)] += 1
#             count += 1
#         if j-1 >= 0:
#             A[ind(i,j), ind(i,j-1)] += 1
#             count += 1
#         if j+1 < ny:
#             A[ind(i,j), ind(i,j+1)] += 1
#             count += 1
#         A[ind(i,j), ind(i,j)] += -count

@ti.kernel
def fillEyeMatrixBuilder(A: ti.sparse_matrix_builder()):
    for i,j in ti.ndrange(nx, ny):
        A[ind(i,j), ind(i,j)] += 1

def buildMatrices():
    fillEyeMatrixBuilder(I_builder)
    return I_builder.build()
 
@ti.kernel
def init():
    for i,j in ti.ndrange(nx, ny):
        if (float(i)-heat_center[0])**2 + (float(j)-heat_center[1])**2 <= heat_radius**2:
            temp[ind(i, j)] = t_max # source
            temp_1[ind(i, j)] = t_max # source
        else:
            temp[ind(i, j)] = t_min
            temp_1[ind(i, j)] = t_min

@ti.kernel
def update_source():
    for i,j in ti.ndrange(nx, ny):
        if (float(i)-heat_center[0])**2 + (float(j)-heat_center[1])**2 <= heat_radius**2:
            temp_1_sm[ind(i, j)] = t_max


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
    for i,j in ti.ndrange(nx, ny):
        for k,l in ti.ndrange(scatter, scatter):
            color[i*scatter+k,j*scatter+l] = get_color(t[ind(i,j)], tmin, tmax)

# GUI
my_gui = ti.GUI("Diffuse", (resx, resy))
my_gui.show()

init()
velocity.fill(ti.Vector([0.0, 0.0]))
diffusivity.fill(1.0)
I = buildMatrices()

for i in range(20000):
    
    for _ in range(substep):
        diffuse(h/substep)
        update_source_and_commit()

    if i % 1000 == 0:
        print(f"Step {i}")
        temperature_to_color(temp_1, pixels, t_min, t_max)
        my_gui.set_image(pixels)
        my_gui.show(f"images/output_{i:05}.png")