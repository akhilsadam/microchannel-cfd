import vtk
import pyvista as pv
import numpy as np
from scipy import interpolate as ip
import jpcm
from tqdm import tqdm

# All the VTR file names
name = '../LB_SinglePhase'
stepsize = 2000
n = 26
keys = ['velocity_0', 'velocity_1', 'velocity_2', 'rho',]
cmp = np.array(jpcm.get('sky').colors)

high_memory = False

filenames = [f'{name}_{i*stepsize}.vtr' for i in range(n)]

def opacity(rho):
    mx = np.max(rho)
    tol = 0.01*(mx - np.min(rho))
    
    return np.where(rho < mx-tol, 0.5, 0)

def get_data(grid,k):
    if '_' in k:
        sk = k.split('_')
        return grid.point_data[sk[0]].T[int(sk[1])], grid.point_data['rho']
    return grid.point_data[k], grid.point_data['rho']
    
reader = vtk.vtkXMLRectilinearGridReader()

def update_reader(fname):
    reader.SetFileName(fname)
    reader.Modified()
    reader.Update()

for k in keys:
    
    update_reader(filenames[0])
    grid = pv.wrap(reader.GetOutput())

    # Create a pl object and set the scalars to the Z height
    pl = pv.Plotter()
    # pl.camera_position = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    # pl.camera.zoom(1.5)
    pl.open_gif(f'{k}.gif')
    
    mn = 0
    mx = 0
    grids = []
    datas = []
    rhos = []
    
    print(f"Initializing {name}")
    for fname in tqdm(filenames):
        update_reader(fname)
        print(f'{fname} is being processed')
        grid = pv.wrap(reader.GetOutput())
        data, rho = get_data(grid,k)
        mn = min(mn, np.min(data))
        mx = max(mx, np.max(data))
        
        if not high_memory:
            grids.append(grid)
            datas.append(data)
            rhos.append(rho)
    
    print(f"Plotting {name}")
    for fname in tqdm(filenames):
        if high_memory:
            update_reader(fname)
            grid = pv.wrap(reader.GetOutput())
            data, rho = get_data(grid,k)
        else:
            grid, data, rho = grids.pop(0), datas.pop(0), rhos.pop(0)   
        
        scalars = (data - mn) / (mx - mn)
        itpdata = ip.pchip_interpolate(np.linspace(0,1,len(cmp)), cmp, scalars)

        scalars = np.concatenate([itpdata, opacity(rho).reshape((-1,1))], axis=1)
        scalars *= 255
        scalars = scalars.astype(np.uint8)
        vol = pl.add_volume(volume=grid, scalars=scalars, name=f'{name}', show_scalar_bar=False)
        vol.prop.interpolation_type = 'linear'
        pl.write_frame()

    # Close movie and delete object
    pl.close()