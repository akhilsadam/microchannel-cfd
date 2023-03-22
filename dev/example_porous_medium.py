import time, os
import taichi as ti

os.environ['TI_DEVICE_MEMORY_FRACTION'] = '0.9'
os.environ['TI_USE_UNIFIED_MEMORY'] = '0'

path = f'{os.getcwd()}/img_ftb131.txt'

ti.init(arch=ti.cuda, kernel_profiler=False, print_ir=False, advanced_optimization=True,fast_math=True)
import MCFD_3D_SinglePhase_Solver as MCFD

time_init = time.time()
time_now = time.time()
time_pre = time.time()             


lb3d = MCFD.MCFD_Solver_Single_Phase(nx=131,ny=131,nz=131, save_images=True)

lb3d.init_geo(path)
lb3d.set_bc_rho_x1(0.99)
lb3d.set_bc_rho_x0(1.0)
lb3d.init_simulation()

for iter in range(50000+1):
    lb3d.step()

    if (iter%500==0):

        time_pre = time_now
        time_now = time.time()
        diff_time = int(time_now-time_pre)
        elap_time = int(time_now-time_init)
        m_diff, s_diff = divmod(diff_time, 60)
        h_diff, m_diff = divmod(m_diff, 60)
        m_elap, s_elap = divmod(elap_time, 60)
        h_elap, m_elap = divmod(m_elap, 60)

        print('----------Time between two outputs is %dh %dm %ds; elapsed time is %dh %dm %ds----------------------' %(h_diff, m_diff, s_diff,h_elap,m_elap,s_elap))
        print('The %dth iteration, Max Force = %f,  force_scale = %f\n\n ' %(iter, 10.0,  10.0))

        if (iter%2000==0):
            lb3d.export_VTK(iter)
            
