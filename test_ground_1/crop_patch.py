import netCDF4
import numpy as np


def read_rs_to_numpy(in_file):
    with netCDF4.Dataset(in_file) as nf:

        rs_01 = nf.variables["albedo_01"][:].data
        rs_02 = nf.variables["albedo_02"][:].data
        rs_03 = nf.variables["albedo_03"][:].data
        rs_04 = nf.variables["albedo_04"][:].data
        rs_05 = nf.variables["albedo_05"][:].data
        rs_06 = nf.variables["albedo_06"][:].data
        rs_07 = nf.variables["tbb_07"][:].data
        rs_08 = nf.variables["tbb_08"][:].data
        rs_09 = nf.variables["tbb_09"][:].data
        rs_10 = nf.variables["tbb_10"][:].data
        rs_11 = nf.variables["tbb_11"][:].data
        rs_12 = nf.variables["tbb_12"][:].data
        rs_13 = nf.variables["tbb_13"][:].data
        rs_14 = nf.variables["tbb_14"][:].data
        rs_15 = nf.variables["tbb_15"][:].data
        rs_16 = nf.variables["tbb_16"][:].data

    hsi =  np.array((rs_01,rs_02, rs_03, rs_04, rs_05, rs_06, rs_07, rs_08,
                    rs_09,rs_10, rs_11, rs_12, rs_13, rs_14, rs_15, rs_16))
    return hsi

def npy_to_arr(root,filecode):
    path = root + "/"+filecode+'.npy'
    whole_hsi = np.load(path)

    return whole_hsi
def crop_patch(whole_hsi,loc_y,loc_x):
    step = 60  # 200 步长
    w = 300  # 1000 长宽
    max = whole_hsi.reshape(16, -1).max(axis=1).reshape(16, 1, 1)
    min = whole_hsi.reshape(16, -1).min(axis=1).reshape(16, 1, 1)
    whole_hsi = (whole_hsi - min) / (max - min) #* 2 - 1
    patch = whole_hsi[:, loc_y * step:loc_y * step + w, loc_x * step:loc_x * step + w]
    return patch

def get_curve(patch, y_per, x_per):

    return patch[:,(int)(300*y_per),(int)(300*x_per)]

if __name__ == '__main__':
    root = '/data/zdxy/DataSets/MLC_16c/small_whole'
    file_code = '20190814'
    whole_hsi = npy_to_arr(root,file_code)
    patch = crop_patch(whole_hsi,5,11)
    curve = get_curve(patch,0.282,0.330)
    print(curve)