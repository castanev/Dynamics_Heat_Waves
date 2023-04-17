# This code is for the collection of data (from NCEP and models outputs) and the construction of netcdf files.
#  - Surface temperature (at 1000 hPa)
#  - u component of wind at 300 hPa
#  - v component of wind at 300 hPa

#import wget
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import datetime as dt
import os
import netCDF4
from scipy import interpolate
import matplotlib.colors as mcolors
from Functions import *


name = 'ERA5'
path_figures = f'{os.path.abspath(os.getcwd())}/{name}/Figures/'
path_outputs = f'{os.path.abspath(os.getcwd())}/{name}/Data/'

path_figures = f'/home/castanev/Heat-waves-dynamics/{name}/Figures'
path_outputs = f'/scratch/brown/castanev/{name}/'


# INPUTS TO CHANGE ===========================================================================================================================
path_data_t = f'/scratch/brown/castanev/NCEP/NCEP_data_T/'
path_data_v = f'/scratch/brown/castanev/NCEP/NCEP_data_v/'
path_data_u = f'/scratch/brown/castanev/NCEP/NCEP_data_u/'

path_data_CAM = '/depot/wanglei/data/CAM_CTRL/'
path_data_LENS = '/depot/wanglei/data/LENS/'
path_data_ERA5 = '/depot/wanglei/data/Reanalysis/ERA5/'
# ============================================================================================================================================



# =================================== NCEP ================================================================
# Originaly, the latitudes in NCEP files are in descending order, the outputs here are in ascending order.
# if name == 'NCEP':
    # T at the surface ============================================================================
    # Saving T at approximately 1000 hPa (.nc) ====================================================
    # years = pd.date_range(start=dt.date(1948, 1,1), end=dt.date(2023, 1, 1), freq='A')

    # # Downloading the data of surface temperature
    # for year in years.strftime('%Y'):
    #     print(f'ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/air.sig995.{year}.nc')
    #     filename = wget.download(f'ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/air.sig995.{year}.nc', out = f'{path_data_t}')

    # ncfile = Dataset(f'{path_data_t}air.sig995.1948.nc')
    # lats = np.array(ncfile['lat'])
    # lons = np.array(ncfile['lon'])


    # new_lats = lats[::-1]
    # list_data = np.sort(os.listdir(f'{path_data_t}'))
    # # Concatenating the data into one array, considering only u in 300hPa
    # contt = 0
    # for i in list_data:
    #     if 'air.sig995.' in i:
    #         contt = contt + 1
    #         datat_i_nc = Dataset(f'{path_data_t}{i}')  #(t, lat, lon)
    #         datat_i = np.array(datat_i_nc['air'])[:, ::-1, :]  #t at surface
    #         timei_t = np.array(datat_i_nc['time']) #Daily
    #         if contt == 1: t = datat_i; dates_t = timei_t
    #         else: t = np.concatenate((t, datat_i), axis = 0); dates_t = np.concatenate((dates_t, timei_t))
    #         print(contt)
    #         print(f'{path_data_t}{i}')


    # nc_name = f't1000_{name}.nc'
    # ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

    # ncfile.createDimension('lat', len(new_lats))
    # ncfile.createDimension('lon', len(lons))
    # ncfile.createDimension('time', len(dates_t))

    # var_lats = ncfile.createVariable('lat', 'f', ('lat'))
    # var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    # var_time = ncfile.createVariable('time', 'f', ('time'))

    # var_lats[:] = new_lats
    # var_lons[:] = lons
    # var_time[:] = dates_t

    # vwnd = ncfile.createVariable('t1000', 'f', ('time', 'lat', 'lon'))
    # vwnd[:, :, :] = t[:, :, :]
    # ncfile.close()


    # # u wind at 300 hPa ===========================================================================
    # #==============================================================================================
    # list_data = np.sort(os.listdir(f'{path_data_u}'))
    # nc_name = f'uwnd.1948.nc'
    # ncfile = Dataset(f'{path_data_u}/{nc_name}')
    # lev = np.array(ncfile['level'])  
    # lats = np.array(ncfile['lat'])  
    # lons = np.array(ncfile['lon'])  
    # pos_lev = np.where(abs(lev-300) == np.min(abs(lev-300)))

    # new_lats = lats[::-1]

    # variable = 'u_300'
    # contt = 0
    # for i in list_data:
    #     if 'uwnd' in i:
    #         contt = contt + 1
    #         datat_i_nc = Dataset(f'{path_data_u}{i}')  #(t, level, lat, lon)
    #         datat_i = np.array(datat_i_nc['uwnd'])[:, pos_lev[0][0], :, :]
    #         datat_i = datat_i[:,::-1,:]

    #         timei = np.array(datat_i_nc['time'])
    #         df = pd.DataFrame(np.reshape(datat_i, (datat_i.shape[0],datat_i.shape[1]*datat_i.shape[2])), index = pd.DatetimeIndex([dt.datetime(1800,1,1) + dt.timedelta(hours = int(timei[i])) for i in range(len(timei))]))

    #         if contt == 1: dataa = datat_i; dates = df.index
    #         else: dataa = np.concatenate((dataa, datat_i), axis = 0); dates = np.concatenate((dates, df.index))


    # # Saving the data
    # # Lats in ascending order 
    # nc_name = f'{variable}_{name}.nc'
    # ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

    # ncfile.createDimension('lat', len(new_lats))
    # ncfile.createDimension('lon', len(lons))
    # ncfile.createDimension('time', len(dates))

    # var_lats = ncfile.createVariable('lat', 'f', ('lat'))
    # var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    # var_time = ncfile.createVariable('time', 'f', ('time'))

    # var_lats[:] = new_lats
    # var_lons[:] = lons
    # var_time[:] = dates

    # varr = ncfile.createVariable(variable, 'f', ('time', 'lat', 'lon'))
    # varr[:, :, :] = dataa[:, :, :]
    # ncfile.close()


    # # v wind at 300 hPa ===========================================================================
    # #==============================================================================================
    # list_data = np.sort(os.listdir(f'{path_data_v}'))
    # nc_name = f'vwnd.1948.nc'
    # ncfile = Dataset(f'{path_data_v}/{nc_name}')
    # lev = np.array(ncfile['level'])  
    # lats = np.array(ncfile['lat'])  
    # lons = np.array(ncfile['lon'])  
    # pos_lev = np.where(abs(lev-300) == np.min(abs(lev-300)))

    # new_lats = lats[::-1]

    # variable = 'v_300'
    # contt = 0
    # for i in list_data:
    #     if 'vwnd' in i:
    #         contt = contt + 1
    #         datat_i_nc = Dataset(f'{path_data_v}{i}')  #(t, level, lat, lon)
    #         datat_i = np.array(datat_i_nc['vwnd'])[:, pos_lev[0][0], :, :]
    #         datat_i = datat_i[:,::-1,:]

    #         timei = np.array(datat_i_nc['time'])
    #         df = pd.DataFrame(np.reshape(datat_i, (datat_i.shape[0],datat_i.shape[1]*datat_i.shape[2])), index = pd.DatetimeIndex([dt.datetime(1800,1,1) + dt.timedelta(hours = int(timei[i])) for i in range(len(timei))]))

    #         if contt == 1: dataa = datat_i; dates = df.index
    #         else: dataa = np.concatenate((dataa, datat_i), axis = 0); dates = np.concatenate((dates, df.index))
    #         print(i)


    # # Saving the data
    # # Lats in ascending order 
    # nc_name = f'{variable}_{name}.nc'
    # ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

    # ncfile.createDimension('lat', len(new_lats))
    # ncfile.createDimension('lon', len(lons))
    # ncfile.createDimension('time', len(dates))

    # var_lats = ncfile.createVariable('lat', 'f', ('lat'))
    # var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    # var_time = ncfile.createVariable('time', 'f', ('time'))

    # var_lats[:] = new_lats
    # var_lons[:] = lons
    # var_time[:] = dates

    # varr = ncfile.createVariable(variable, 'f', ('time', 'lat', 'lon'))
    # varr[:, :, :] = dataa[:, :, :]
    # ncfile.close()


    # # Downloading the data of surface temperature
    # for year in years.strftime('%Y'):
    #     print(f'{year}.nc')
    #     filename = wget.download(f'ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/air.2m.gauss.{year}.nc', out = f'{path_data_t}')

    # ncfile = Dataset(f'{path_data_t}air.2m.gauss.1948.nc')
    # lats = np.array(ncfile['lat'])
    # lons = np.array(ncfile['lon'])

    # new_lats = lats[::-1]
    # list_data = np.sort(os.listdir(f'{path_data_t}'))
    # # Concatenating the data into one array, considering only u in 300hPa
    # contt = 0
    # for i in list_data:
    #     if 'air.2m.' in i:
    #         contt = contt + 1
    #         datat_i_nc = Dataset(f'{path_data_t}{i}')  #(t, lat, lon)
    #         datat_i = np.array(datat_i_nc['air'])[:, ::-1, :]  #t at surface
    #         timei_t = np.array(datat_i_nc['time']) #Daily
    #         if contt == 1: t = datat_i; dates_t = timei_t
    #         else: t = np.concatenate((t, datat_i), axis = 0); dates_t = np.concatenate((dates_t, timei_t))
    #         print(contt)
    #         print(f'{path_data_t}{i}')


    # nc_name = f't2m_{name}.nc'
    # ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

    # ncfile.createDimension('lat', len(new_lats))
    # ncfile.createDimension('lon', len(lons))
    # ncfile.createDimension('time', len(dates_t))

    # var_lats = ncfile.createVariable('lat', 'f', ('lat'))
    # var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    # var_time = ncfile.createVariable('time', 'f', ('time'))

    # var_lats[:] = new_lats
    # var_lons[:] = lons
    # var_time[:] = dates_t

    # vwnd = ncfile.createVariable('t1000', 'f', ('time', 'lat', 'lon'))
    # vwnd[:, :, :] = t[:, :, :]
    # ncfile.close()



# =================================== CAM ================================================================
# if name == 'CAM':
#     list_data = np.sort(os.listdir(f'{path_data_CAM}'))

#     # Reading lats and lons of t at 1000 hPa, globally (CAM)
#     nc_name = f'CAM_CTRL.cam.h1.0061-05-01-00000.nc'
#     ncfile = Dataset(f'{path_data_CAM}{nc_name}')
#     lats = np.array(ncfile['lat'])  # 121. Res = 1.89°
#     lons = np.array(ncfile['lon'])  # 144. Res = 2.5°
#     lev = np.array(ncfile['lev'])  # lev[-1] = 992.5561. Surface
#     pos_lev = np.where(abs(lev-300) == np.min(abs(lev-300)))

#     # T at the surface ============================================================================
#     # Saving T at approximately 1000 hPa (.nc) ====================================================
#     # Lats in ascending order 
#     contt = 0
#     for i in list_data:
#         if 'CAM_CTRL.cam.h1' in i:
#             contt = contt + 1
#             datat_i_nc = Dataset(f'{path_data_CAM}{i}')  #(t, level, lat, lon) 
#             datat_i = np.array(datat_i_nc['T'])[:, -1, :, :]  #t at surface
#             timei_t = datat_i_nc.variables['time']
#             timei_t = netCDF4.num2date(timei_t[:], timei_t.units, timei_t.calendar)
#             timei_t_str = [timei_t[i].strftime('%Y%m%d') for i in range(len(timei_t))]
#             if contt == 1: t = datat_i; dates_t = timei_t_str
#             else: t = np.concatenate((t, datat_i), axis = 0); dates_t = np.concatenate((dates_t, timei_t_str))
#             print(f'{path_data_CAM}{i}')


#     nc_name = f't1000_{name}.nc'
#     ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

#     ncfile.createDimension('lat', len(lats))
#     ncfile.createDimension('lon', len(lons))
#     ncfile.createDimension('time', len(dates_t))

#     var_lats = ncfile.createVariable('lat', 'f', ('lat'))
#     var_lons = ncfile.createVariable('lon', 'f', ('lon'))
#     var_time = ncfile.createVariable('time', 'f', ('time'))

#     var_lats[:] = lats
#     var_lons[:] = lons
#     var_time[:] = dates_t

#     vwnd = ncfile.createVariable('t1000', 'f', ('time', 'lat', 'lon'))
#     vwnd[:, :, :] = t[:, :, :]
#     ncfile.close()


#     # V 300 =======================================================================================
#     # Saving v at 300 hPa hPa (.nc) ===============================================================
#     # Lats in ascending order 
#     contt = 0
#     for i in list_data:
#         if 'CAM_CTRL.cam.h1' in i:
#             contt = contt + 1
#             datat_i_nc = Dataset(f'{path_data_CAM}{i}')  #(t, level, lat, lon) 
#             datat_i = np.array(datat_i_nc['V'])[:, pos_lev, :, :]  #t at surface
#             timei_t = datat_i_nc.variables['time']
#             timei_t = netCDF4.num2date(timei_t[:], timei_t.units, timei_t.calendar)
#             timei_t_str = [timei_t[i].strftime('%Y%m%d') for i in range(len(timei_t))]
#             if contt == 1: t = datat_i; dates_t = timei_t_str
#             else: t = np.concatenate((t, datat_i), axis = 0); dates_t = np.concatenate((dates_t, timei_t_str))
#             print(f'{path_data_CAM}{i}')


#     nc_name = f'v_300_{name}.nc'
#     ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

#     ncfile.createDimension('lat', len(lats))
#     ncfile.createDimension('lon', len(lons))
#     ncfile.createDimension('time', len(dates_t))

#     var_lats = ncfile.createVariable('lat', 'f', ('lat'))
#     var_lons = ncfile.createVariable('lon', 'f', ('lon'))
#     var_time = ncfile.createVariable('time', 'f', ('time'))

#     var_lats[:] = lats
#     var_lons[:] = lons
#     var_time[:] = dates_t

#     vwnd = ncfile.createVariable('v_300', 'f', ('time', 'lat', 'lon'))
#     vwnd[:, :, :] = t[:, :, :]
#     ncfile.close()


#     # U 300 =======================================================================================
#     # Saving u at 300 hPa hPa (.nc) ===============================================================
#     # Lats in ascending order 
#     contt = 0
#     for i in list_data:
#         if 'CAM_CTRL.cam.h1' in i:
#             contt = contt + 1
#             datat_i_nc = Dataset(f'{path_data_CAM}{i}')  #(t, level, lat, lon) 
#             datat_i = np.array(datat_i_nc['U'])[:, pos_lev, :, :]  #t at surface
#             timei_t = datat_i_nc.variables['time']
#             timei_t = netCDF4.num2date(timei_t[:], timei_t.units, timei_t.calendar)
#             timei_t_str = [timei_t[i].strftime('%Y%m%d') for i in range(len(timei_t))]
#             if contt == 1: t = datat_i; dates_t = timei_t_str
#             else: t = np.concatenate((t, datat_i), axis = 0); dates_t = np.concatenate((dates_t, timei_t_str))
#             print(f'{path_data_CAM}{i}')

#     nc_name = f'u_300_{name}.nc'
#     ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

#     ncfile.createDimension('lat', len(lats))
#     ncfile.createDimension('lon', len(lons))
#     ncfile.createDimension('time', len(dates_t))

#     var_lats = ncfile.createVariable('lat', 'f', ('lat'))
#     var_lons = ncfile.createVariable('lon', 'f', ('lon'))
#     var_time = ncfile.createVariable('time', 'f', ('time'))

#     var_lats[:] = lats
#     var_lons[:] = lons
#     var_time[:] = dates_t

#     vwnd = ncfile.createVariable('u_300', 'f', ('time', 'lat', 'lon'))
#     vwnd[:, :, :] = t[:, :, :]
#     ncfile.close()


# # =================================== LENS ================================================================
# if name == 'LENS':
#     list_data = np.sort(os.listdir(f'{path_data_LENS}TS/'))

#     # Reading lats and lons of t at 1000 hPa, globally (CAM)
#     nc_name = f'b.e11.B1850C5CN.f09_g16.005.cam.h1.TS.04020101-04991231.nc'
#     ncfile = Dataset(f'{path_data_LENS}TS/{nc_name}')
#     lats = np.array(ncfile['lat'])  # 192. Res = 0.94°
#     lons = np.array(ncfile['lon'])  # 288. Res = 1.25°
#     pos_18 = np.where(abs(lats-18) == np.min(abs(lats-18)))[0][0]
#     pos_70 = np.where(abs(lats-70) == np.min(abs(lats-70)))[0][0]
#     lats_midlat = lats[pos_18:pos_70+1]
    
#     # nc_name_CAM = f'CAM_CTRL.cam.h1.0061-05-01-00000.nc'
#     # ncfile_CAM = Dataset(f'{path_data_CAM}{nc_name_CAM}')
#     # lats_CAM = np.array(ncfile['lat'])  # 121. Res = 1.89°
#     # lons_CAM = np.array(ncfile['lon'])  # 144. Res = 2.5°
#     # pos_18_CAM = np.where(abs(lats_CAM-18) == np.min(abs(lats_CAM-18)))[0][0]
#     # pos_70_CAM = np.where(abs(lats_CAM-70) == np.min(abs(lats_CAM-70)))[0][0]
#     # lats_midlat_CAM = lats_CAM[pos_18:pos_70+1]

#     # T at the surface ============================================================================
#     # Saving T at approximately 1000 hPa (.nc) ====================================================
#     # Lats in ascending order 
#     contt = 0
#     for i in list_data:
#         if 'b.e11.B1850C5CN.f09_g16.005.cam.h1.TS.' in i:
#             contt = contt + 1
#             datat_i_nc = Dataset(f'{path_data_LENS}TS/{i}')  #(t, lat, lon) 
#             timei_t = datat_i_nc.variables['time']
#             timei_t = netCDF4.num2date(timei_t[:], timei_t.units, timei_t.calendar)
#             timei_t_str = [timei_t[i].strftime('%Y%m%d') for i in range(len(timei_t))]
#             datat_i = np.array(datat_i_nc['TS'])[:, pos_18:pos_70+1, :]

#             # # Interpolation to reduce the shape
#             # datat_i = np.zeros(([len(timei_t_str), len(lats_midlat_CAM), len(lons_CAM)]))
#             # for ttt in range(len(timei_t_str)):
#             #     f_interpolate = interpolate.interp2d(lons, lats_midlat, np.array(datat_i_nc['TS'])[ttt, pos_18:pos_70+1, :], kind='cubic')
#             #     datat_i[ttt,:,:] = f_interpolate(lons_CAM, lats_midlat_CAM)
                
#             if contt == 1: t = datat_i; dates_t = timei_t_str
#             else: t = np.concatenate((t, datat_i), axis = 0); dates_t = np.concatenate((dates_t, timei_t_str))
#             print(f'{i}')
            

    
#     nc_name = f't1000_{name}.nc'
#     ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

#     ncfile.createDimension('lat', len(lats_midlat))
#     ncfile.createDimension('lon', len(lons))
#     ncfile.createDimension('time', len(dates_t))

#     var_lats = ncfile.createVariable('lat', 'f', ('lat'))
#     var_lons = ncfile.createVariable('lon', 'f', ('lon'))
#     var_time = ncfile.createVariable('time', 'f', ('time'))

#     var_lats[:] = lats_midlat
#     var_lons[:] = lons
#     var_time[:] = dates_t

#     vwnd = ncfile.createVariable('t1000', 'f', ('time', 'lat', 'lon'))
#     vwnd[:, :, :] = t[:, :, :]
#     ncfile.close()


#     # V 200 =======================================================================================
#     # Saving v at 200 hPa hPa (.nc) ===============================================================
#     # Lats in ascending order 
#     list_data = np.sort(os.listdir(f'{path_data_LENS}SF200/'))

#     contt = 0
#     for i in list_data:
#         if 'b.e11.B1850C5CN.f09_g16.005.cam.h1.V200.' in i:
#             contt = contt + 1
#             datat_i_nc = Dataset(f'{path_data_LENS}SF200/{i}')  #(t, level, lat, lon) 
#             timei_t = datat_i_nc.variables['time']
#             timei_t = netCDF4.num2date(timei_t[:], timei_t.units, timei_t.calendar)
#             timei_t_str = [timei_t[i].strftime('%Y%m%d') for i in range(len(timei_t))]
#             datat_i = np.array(datat_i_nc['V200'])[:, pos_18:pos_70+1, :]

#             # # Interpolation to reduce the shape
#             # datat_i = np.zeros(([len(timei_t_str), len(lats_midlat_CAM), len(lons_CAM)]))
#             # for ttt in range(len(timei_t_str)):
#             #     f_interpolate = interpolate.interp2d(lons, lats_midlat, np.array(datat_i_nc['V200'])[ttt, pos_18:pos_70+1, :], kind='cubic')
#             #     datat_i[ttt,:,:] = f_interpolate(lons_CAM, lats_midlat_CAM)
                
#             if contt == 1: t = datat_i; dates_t = timei_t_str
#             else: t = np.concatenate((t, datat_i), axis = 0); dates_t = np.concatenate((dates_t, timei_t_str))
#             print(f'{i}')


#     nc_name = f'v_200_{name}.nc'
#     ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

#     ncfile.createDimension('lat', len(lats_midlat))
#     ncfile.createDimension('lon', len(lons))
#     ncfile.createDimension('time', len(dates_t))

#     var_lats = ncfile.createVariable('lat', 'f', ('lat'))
#     var_lons = ncfile.createVariable('lon', 'f', ('lon'))
#     var_time = ncfile.createVariable('time', 'f', ('time'))

#     var_lats[:] = lats_midlat
#     var_lons[:] = lons
#     var_time[:] = dates_t

#     vwnd = ncfile.createVariable('v_300', 'f', ('time', 'lat', 'lon'))
#     vwnd[:, :, :] = t[:, :, :]
#     ncfile.close()


#     # SF 200 =======================================================================================
#     # Saving sf at 200 hPa hPa (.nc) ===============================================================
#     # Lats in ascending order 
#     contt = 0
#     for i in list_data:
#         if 'b.e11.B1850C5CN.f09_g16.005.cam.h1.SF200.' in i:
#             contt = contt + 1
#             datat_i_nc = Dataset(f'{path_data_LENS}SF200/{i}')  #(t, level, lat, lon) 
#             timei_t = datat_i_nc.variables['time']
#             timei_t = netCDF4.num2date(timei_t[:], timei_t.units, timei_t.calendar)
#             timei_t_str = [timei_t[i].strftime('%Y%m%d') for i in range(len(timei_t))]
#             datat_i = np.array(datat_i_nc['SF200'])[:, pos_18:pos_70+1, :]

#             # # Interpolation to reduce the shape
#             # datat_i = np.zeros(([len(timei_t_str), len(lats_midlat_CAM), len(lons_CAM)]))
#             # for ttt in range(len(timei_t_str)):
#             #     f_interpolate = interpolate.interp2d(lons, lats_midlat, np.array(datat_i_nc['SF200'])[ttt, pos_18:pos_70+1, :], kind='cubic')
#             #     datat_i[ttt,:,:] = f_interpolate(lons_CAM, lats_midlat_CAM)
                
#             if contt == 1: t = datat_i; dates_t = timei_t_str
#             else: t = np.concatenate((t, datat_i), axis = 0); dates_t = np.concatenate((dates_t, timei_t_str))
#             print(f'{i}')

#     nc_name = f'sf_200_{name}.nc'
#     ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

#     ncfile.createDimension('lat', len(lats_midlat))
#     ncfile.createDimension('lon', len(lons))
#     ncfile.createDimension('time', len(dates_t))

#     var_lats = ncfile.createVariable('lat', 'f', ('lat'))
#     var_lons = ncfile.createVariable('lon', 'f', ('lon'))
#     var_time = ncfile.createVariable('time', 'f', ('time'))

#     var_lats[:] = lats_midlat
#     var_lons[:] = lons
#     var_time[:] = dates_t

#     vwnd = ncfile.createVariable('SF', 'f', ('time', 'lat', 'lon'))
#     vwnd[:, :, :] = t[:, :, :]
#     ncfile.close()





# =================================== ERA5 ================================================================
# Originaly, the latitudes in NCEP files are in descending order, the outputs here are in ascending order.
if name == 'ERA5':
    # T at the surface ============================================================================
    # Saving T at approximately 1000 hPa (.nc) ====================================================
    ncfile = Dataset(f'{path_data_ERA5}TS/TS_1950.nc')
    lats = np.array(ncfile['latitude'])
    lons = np.array(ncfile['longitude'])

    new_lats = lats[::-1]
    list_data = np.sort(os.listdir(f'{path_data_ERA5}TS/'))
    # Concatenating the data into one array, considering only u in 300hPa
    contt = 0
    for i in list_data:
        contt = contt + 1
        datat_i_nc = Dataset(f'{path_data_ERA5}TS/{i}')  #(t, lat, lon)
        datat_i = np.array(datat_i_nc['t'])[:, ::-1, :]  #t at surface
        timei_t = np.array(datat_i_nc['time']) #Daily, hours since 1900-01-01 00:00:00.0
        if contt == 1: t = datat_i; dates_t = timei_t
        else: t = np.concatenate((t, datat_i), axis = 0); dates_t = np.concatenate((dates_t, timei_t))
        print(contt)
        print(f'{path_data_t}{i}')


    nc_name = f't1000_{name}.nc'
    ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

    ncfile.createDimension('lat', len(new_lats))
    ncfile.createDimension('lon', len(lons))
    ncfile.createDimension('time', len(dates_t))

    var_lats = ncfile.createVariable('lat', 'f', ('lat'))
    var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    var_time = ncfile.createVariable('time', 'f', ('time'))

    var_lats[:] = new_lats
    var_lons[:] = lons
    var_time[:] = dates_t

    vwnd = ncfile.createVariable('t1000', 'f', ('time', 'lat', 'lon'))
    vwnd[:, :, :] = t[:, :, :]
    ncfile.close()


    # u wind at 300 hPa ===========================================================================
    #==============================================================================================
    list_data = np.sort(os.listdir(f'{path_data_ERA5}uv_wind/'))
    nc_name = f'uv_wind_1950.nc'
    ncfile = Dataset(f'{path_data_ERA5}/uv_wind/{nc_name}') 
    lats = np.array(ncfile['latitude'])  
    lons = np.array(ncfile['longitude'])  

    new_lats = lats[::-1]

    variable = 'u_300'
    contt = 0
    for i in list_data:
        contt = contt + 1
        datat_i_nc = Dataset(f'{path_data_ERA5}uv_wind/{i}')  #(t, level, lat, lon)
        datat_i = np.array(datat_i_nc['u'])[:, ::-1, :]

        timei = np.array(datat_i_nc['time'])
        df = pd.DataFrame(np.reshape(datat_i, (datat_i.shape[0],datat_i.shape[1]*datat_i.shape[2])), index = pd.DatetimeIndex([dt.datetime(1800,1,1) + dt.timedelta(hours = int(timei[i])) for i in range(len(timei))]))

        if contt == 1: dataa = datat_i; dates = df.index
        else: dataa = np.concatenate((dataa, datat_i), axis = 0); dates = np.concatenate((dates, df.index))


    # Saving the data
    # Lats in ascending order 
    nc_name = f'{variable}_{name}.nc'
    ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

    ncfile.createDimension('lat', len(new_lats))
    ncfile.createDimension('lon', len(lons))
    ncfile.createDimension('time', len(dates))

    var_lats = ncfile.createVariable('lat', 'f', ('lat'))
    var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    var_time = ncfile.createVariable('time', 'f', ('time'))

    var_lats[:] = new_lats
    var_lons[:] = lons
    var_time[:] = dates

    varr = ncfile.createVariable(variable, 'f', ('time', 'lat', 'lon'))
    varr[:, :, :] = dataa[:, :, :]
    ncfile.close()


    # v wind at 300 hPa ===========================================================================
    #==============================================================================================
    list_data = np.sort(os.listdir(f'{path_data_ERA5}uv_wind/'))
    nc_name = f'uv_wind_1950.nc'
    ncfile = Dataset(f'{path_data_ERA5}/uv_wind/{nc_name}') 
    lats = np.array(ncfile['latitude'])  
    lons = np.array(ncfile['longitude'])  

    new_lats = lats[::-1]

    variable = 'v_300'
    contt = 0
    for i in list_data:
        contt = contt + 1
        datat_i_nc = Dataset(f'{path_data_ERA5}uv_wind/{i}')  #(t, level, lat, lon)
        datat_i = np.array(datat_i_nc['v'])[:, :, :]
        datat_i = datat_i[:,::-1,:]

        timei = np.array(datat_i_nc['time'])
        df = pd.DataFrame(np.reshape(datat_i, (datat_i.shape[0],datat_i.shape[1]*datat_i.shape[2])), index = pd.DatetimeIndex([dt.datetime(1800,1,1) + dt.timedelta(hours = int(timei[i])) for i in range(len(timei))]))

        if contt == 1: dataa = datat_i; dates = df.index
        else: dataa = np.concatenate((dataa, datat_i), axis = 0); dates = np.concatenate((dates, df.index))
        print(i)


    # Saving the data
    # Lats in ascending order 
    nc_name = f'{variable}_{name}.nc'
    ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

    ncfile.createDimension('lat', len(new_lats))
    ncfile.createDimension('lon', len(lons))
    ncfile.createDimension('time', len(dates))

    var_lats = ncfile.createVariable('lat', 'f', ('lat'))
    var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    var_time = ncfile.createVariable('time', 'f', ('time'))

    var_lats[:] = new_lats
    var_lons[:] = lons
    var_time[:] = dates

    varr = ncfile.createVariable(variable, 'f', ('time', 'lat', 'lon'))
    varr[:, :, :] = dataa[:, :, :]
    ncfile.close()


