# This code is to calculate the phase speed using barotropic average for NCEP, CAM and Dry Core GCM

import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import cartopy
from cartopy import crs
import matplotlib.ticker as mticker
import os
import matplotlib.colors as mcolors
import pandas as pd
import datetime as dt
from netCDF4 import Dataset
import re
from scipy import ndimage
from scipy import interpolate
from Functions import *


# INPUTS TO CHANGE ===========================================================================================================================
# var: the variable name in the nc file. 't1000' for NCEP, 'temp' for the dry core GCM 
# delete_years is the number of years to cut (only for output models). 

path_data_NCEP = '/scratch/brown/castanev/NCEP/NCEP_data_u/'
var_u_NCEP = 'uwnd'

path_data_CAM ='/depot/wanglei/data/CAM_CTRL/'
nc_name_CAM = 'CAM_CTRL.cam.h1.0001-01-01-00000.nc'
var_u_CAM = 'U'

path_data_GCM ='/scratch/brown/castanev/DryCore_Wu/output/'
exp_GCM = ['exp1_Held_Suarez', 'exp2_NCEPsymm_noSeason_noTop', 'exp3_NCEPasymm_noSeason_noTop']
exp_names_fig = ['CTR','ZOB','ROB']
nc_name_GCM = '00000000.atmos_daily.nc'
var_u_GCM = 'ucomp'

delete_years = ''
# ============================================================================================================================================

# path_figures = f'{os.path.abspath(os.getcwd())}/Phase_speed/'
path_figures = f'/home/castanev/Heat-waves-dynamics/Phase_speed/'

names = []
midlat = []
U = []
U_1d = {}

# ======================================================================================================================================
# ============================================================== NCEP ==================================================================
name = 'NCEP'
ncfile = Dataset(f'{path_data_NCEP}uwnd.1948.nc')
lev = np.array(ncfile['level'])
lats = np.array(ncfile['lat'])
lons = np.array(ncfile['lon'])

list_data = np.sort(os.listdir(f'{path_data_NCEP}'))
contt = 0
u = np.zeros([len(list_data),len(lev),len(lats), len(lons)]) * np.NaN
# for i in list_data:
#     if 'uwnd.' in i:
#         contt = contt + 1
#         datat_i_nc = Dataset(f'{path_data_NCEP}{i}')  #(u, level, lat, lon)
#         datat_i = np.mean(np.array(datat_i_nc['uwnd'])[:, :, :, :], axis = 0)
#         u[contt-1,:,:,:] = datat_i
#         print(f'{path_data_NCEP}{i}')

for i in list_data:
    if 'uwnd.' in i:
        contt = contt + 1
        datat_i_nc = Dataset(f'{path_data_NCEP}{i}')  #(u, level, lat, lon)
        timei_t = np.array(datat_i_nc['time']) #Daily
        dates_i = np.array([dt.datetime(1800, 1, 1) + dt.timedelta(hours=int(timei_t[i])) for i in range(len(timei_t))])
        Month_i = np.array([ii.month for ii in dates_i])
        pos_summer = np.where([ii in [6, 7, 8] for ii in Month_i])[0]
        datat_i = np.mean(np.array(datat_i_nc['uwnd'])[pos_summer, :, :, :], axis = 0)
        u[contt-1,:,:,:] = datat_i
        print(f'{path_data_NCEP}{i}')
        # if f'{path_data_NCEP}{i}' == f'{path_data_NCEP}{list_data[1]}': break
        

u = u[:,:,::-1,:] # Lats in ascending order
lats = lats[::-1]
pos_lats_NH_NCEP = np.where(lats>=0)
lats_NH_NCEP = lats[pos_lats_NH_NCEP]
temp_mean = np.nanmean(u, axis=0)

norm = mcolors.DivergingNorm(vmin=-10, vmax = 21, vcenter=0)
li, ls, intervalos, limite, color = -21, 21, 15, 1, 'Spectral_r'
bounds = np.round(np.linspace(li, ls, intervalos), 3)
colormap = center_white_anom(color, intervalos, bounds, limite)
maps_vert(lats, lev, -10, 21, np.mean(temp_mean, axis = 2), colormap, f'{path_figures}U_vertical_{name}.jpg', norm, units='U [m/s]')
maps_vert(lats, lev, -10, 21, np.mean(temp_mean, axis = 2), colormap, f'{path_figures}U_vertical_{name}.svg', norm, units='U [m/s]')


# Density of each layer
densities = []
u_baroc1 = np.zeros_like(temp_mean) * np.nan
for i, l in enumerate(lev):
    rho_i = l / 1000
    densities.append(rho_i)
    u_baroc1[i,:,:] = rho_i * temp_mean[i,:,:]

u_baroc = np.sum(u_baroc1, axis=0)/np.array(densities).sum()

norm = mcolors.DivergingNorm(vmin=-10, vmax = 21, vcenter=0)
li, ls, intervalos, limite, color = -21, 21, 15, 1, 'Spectral_r'
bounds = np.round(np.linspace(li, ls, intervalos), 3)
colormap = center_white_anom(color, intervalos, bounds, limite)
maps1(lons, lats, -10, 21, u_baroc, colormap, f'{path_figures}U_barotropic_average_2D_{name}.jpg', norm, units='barotropic average U [m/s]', topography = True)
maps1(lons, lats, -10, 21, u_baroc, colormap, f'{path_figures}U_barotropic_average_2D_{name}.svg', norm, units='barotropic average U [m/s]', topography = True)

pos_max = np.where((np.nanmean(u_baroc[int(np.floor(len(lats) / 2)):], axis=1) == np.nanmean(
       u_baroc[int(np.floor(len(lats) / 2)):], axis=1).max()))[0][0] + int(np.floor(len(lats) / 2))
#zonal_mean_1D(u_baroc, np.round(lats,1), pos_max, path_figures + f'zonal_mean_U_barotropic_average_{name}.jpg')

u_baroc_1d = np.nanmean(u_baroc, axis=1)
U_1d[name] = u_baroc_1d[pos_lats_NH_NCEP]

U.append(round(np.nanmean(u_baroc[int(np.floor(len(lats)/2)):], axis = 1).max(), 2))
midlat.append(round(lats[pos_max], 2))
names.append(name)


# ======================================================================================================================================
# ============================================================== DRY GCM ===============================================================
delete_years = 3

exp_num = 0
for exp, name in zip(exp_GCM, exp_names_fig):
    list_data = np.sort(os.listdir(f'{path_data_GCM}{exp}/post_processed/output/'))
    if name == 'CTR': list_data = list_data[:-6]
    else: list_data = list_data[:-3]
    list_begindates = np.unique(np.sort(np.array([re.findall(r'\d+', list_data[i][:13])[0] for i in range(len(list_data))]).astype(int)))
    list_begindates = list_begindates[delete_years:]
    # Reading lats and lons of t at 1000 hPa, globally (CAM)
    ncfile = Dataset(f'{path_data_GCM}{exp}/post_processed/output/{nc_name_GCM}')
    lats = np.array(ncfile['lat'])   # 96. Res = 1.85°
    pos_lats_NH = np.where(lats>=0)
    lats_NH = lats[pos_lats_NH]
    lons = np.array(ncfile['lon'])   # 192. Res = 1.875°
    lev = np.array(ncfile['lev'])    # lev[-1] = 987.47. Surface

    u = np.zeros([len(list_data),len(lev), len(lats), len(lons)]) * np.NaN
    contt = 0
    for i in list_begindates: 
        if f'000000{str(i)}.atmos_daily.nc' in list_data:  
            contt = contt + 1    
            data_i_nc = Dataset(f'{path_data_GCM}{exp}/post_processed/output/000000{str(i)}.atmos_daily.nc')
            datat_i = np.mean(np.array(data_i_nc[var_u_GCM])[:,:,:,:], axis=0)  #(time, pfull, lat, lon) 
            u[contt - 1, :, :, :] = datat_i
            print(f'000000{str(i)}.atmos_daily.nc')
            # if f'{path_data_GCM}{exp}/post_processed/output/000000{str(i)}.atmos_daily.nc' == f'{path_data_GCM}{exp}/post_processed/output/000000{str(list_begindates[1])}.atmos_daily.nc': break

    u = u[:,:,:,:]
    temp_mean = np.nanmean(u, axis=0)

    # norm = mcolors.DivergingNorm(vmin=-10, vmax = 21, vcenter=0)
    # li, ls, intervalos, limite, color = -21, 21, 15, 1, 'Spectral_r'
    # bounds = np.round(np.linspace(li, ls, intervalos), 3)
    # colormap = center_white_anom(color, intervalos, bounds, limite)
    # maps_vert(lats, lev, -10, 21, np.mean(temp_mean, axis = 2), colormap, f'{path_figures}U_vertical_{name}.jpg', norm, units='U [m/s]')
    # maps_vert(lats, lev, -10, 21, np.mean(temp_mean, axis = 2), colormap, f'{path_figures}U_vertical_{name}.svg', norm, units='U [m/s]')

    # Density of each layer
    densities = []
    u_baroc1 = np.zeros_like(temp_mean) * np.nan
    for i, l in enumerate(lev):
        rho_i = l / 1000
        densities.append(rho_i)
        u_baroc1[i,:,:] = rho_i * temp_mean[i,:,:]


    u_baroc = np.sum(u_baroc1, axis=0)/np.array(densities).sum()

    # norm = mcolors.DivergingNorm(vmin=-10, vmax = 21, vcenter=0)
    # li, ls, intervalos, limite, color = -21, 21, 15, 1, 'Spectral_r'
    # bounds = np.round(np.linspace(li, ls, intervalos), 3)
    # colormap = center_white_anom(color, intervalos, bounds, limite)
    # maps1(lons, lats, -10, 21, u_baroc, colormap, f'{path_figures}U_barotropic_average_2D_{name}.jpg', norm, units='Barotropic average U [m/s]', topography = False)
    # maps1(lons, lats, -10, 21, u_baroc, colormap, f'{path_figures}U_barotropic_average_2D_{name}.svg', norm, units='Barotropic average U [m/s]', topography = False)
   
    pos_max = np.where((np.nanmean(u_baroc[int(np.floor(len(lats) / 2)):], axis=1) == np.nanmean(
            u_baroc[int(np.floor(len(lats) / 2)):], axis=1).max()))[0][0] + int(np.floor(len(lats) / 2))
    #zonal_mean_1D(u_baroc, np.round(lats,1), pos_max, path_figures + f'zonal_mean_U_barotropic_average_{name}.jpg')

    u_baroc_1d = np.nanmean(u_baroc, axis=1)
    U_1d[name] = u_baroc_1d[pos_lats_NH]

    U.append(round(np.nanmean(u_baroc[int(np.floor(len(lats)/2)):], axis = 1).max(), 2))
    midlat.append(round(lats[pos_max], 2))
    names.append(name)
    exp_num = exp_num + 1


# Comparing jets 
U_1d_NCEP_interp = interpolate.interp1d(lats_NH_NCEP,U_1d['NCEP'])
U_1d['NCEP'] = U_1d_NCEP_interp(lats_NH)
lines = ['-', '-', '-', '--']
colors = ['k', 'grey', 'dodgerblue', 'dodgerblue']
widths = [1.8, 1.3, 1.2, 1.2]
fig = plt.figure(figsize=[7, 5])
# fig = plt.figure(figsize=[11.5/2.54,9.5/2.54])
for name, u_1d, line, color, width in zip(names, U_1d.keys(), lines, colors, widths):
    plot = plt.plot(U_1d[u_1d], color=color, ls = line, label = name, linewidth=width)
#ndimage.rotate(plot, 90)
plt.legend(loc='best')
plt.ylabel('Mean zonal wind [m/s]', fontsize=12)
plt.xlabel('Latitude [°]', fontsize=12)
plt.xticks(np.floor(np.linspace(0, len(lats_NH)-1, 5)).astype(int), np.round(lats_NH[np.floor(np.linspace(0, len(lats_NH)-1, 5)).astype(int)],0), fontsize=12)
plt.yticks(fontsize=12)
#plt.axvspan(25, 50, alpha=0.5, color='dimgray')
plt.savefig(path_figures + f'Jet_all_barotropic_average.jpg', dpi=500)
plt.savefig(path_figures + f'Jet_all_barotropic_average.svg', dpi=500)
plt.close()



# ======================================================================================================================================
# ============================================================== CAM ===================================================================
# delete_years = 3
# name = 'CAM'

# list_data = np.sort(os.listdir(f'{path_data_CAM}'))

# # Reading lats and lons, globally (CAM)
# ncfile = Dataset(f'{path_data_CAM}{nc_name_CAM}')
# lats = np.array(ncfile['lat'])  # 121. Res = 1.89°
# lons = np.array(ncfile['lon'])  # 144. Res = 2.5°
# lev = np.array(ncfile['lev'])   # lev[-1] = 992.5561. Surface

# u = np.zeros([len(list_data),len(lev),len(lats), len(lons)]) * np.NaN
# contt = 0
# for i in list_data:
#     if 'CAM_CTRL.cam.h1' in i:
#         contt = contt + 1
#         datat_i_nc = Dataset(f'{path_data_CAM}{i}')  #(t, level, lat, lon)
#         datat_i = np.mean(np.array(datat_i_nc['U'])[:, :, :, :], axis=0)  #t at surface
#         u[contt - 1, :, :, :] = datat_i
#         print(f'{path_data_CAM}{i}')

# u = u[delete_years:,:,:,:]
# temp_mean = np.nanmean(u, axis=0)

# # Density of each layer
# densities = []
# u_baroc1 = np.zeros_like(temp_mean) * np.nan
# for i, l in enumerate(lev):
#     rho_i = l / 1000
#     densities.append(rho_i)
#     u_baroc1[i,:,:] = rho_i * temp_mean[i,:,:]

# u_baroc = np.sum(u_baroc1, axis=0)/np.array(densities).sum()

# norm = mcolors.DivergingNorm(vmin=u_baroc.min(), vmax = u_baroc.max(), vcenter=0)
# maps1(lons, lats, u_baroc.min(), u_baroc.max(), u_baroc, 'Spectral_r', f'{path_figures}U_barotropic_average_2D_{name}.jpg', norm, units='Barotropic average U [m/s]', topography = True)

# pos_max = np.where((np.nanmean(u_baroc[int(np.floor(len(lats) / 2)):], axis=1) == np.nanmean(
#         u_baroc[int(np.floor(len(lats) / 2)):], axis=1).max()))[0][0] + int(np.floor(len(lats) / 2))
# zonal_mean_1D(u_baroc, np.round(lats,1), pos_max, path_figures + f'zonal_mean_U_barotropic_average_{name}.jpg')

# U.append(round(np.nanmean(u_baroc[int(np.floor(len(lats)/2)):], axis = 1).max(), 2))
# midlat.append(round(lats[pos_max], 2))
# names.append(name)


# ======================================================================================================================================
# ============================================================ PHASE SPEED =============================================================
R = 6367000   #Radius of the Earth [m]
omega = 2*np.pi/(3600*24)  #[rad/s]
wavenumbers = [4, 5, 6, 7]

# Phase speed
colors = ['darkslateblue', 'royalblue', 'mediumpurple', 'skyblue', 'silver']
shapes = ['o', 'x', 'D', '*', '^']
sizes = [40,40,40,50,50]
fig = plt.figure(figsize=[7,5])
# fig = plt.figure(figsize=[11.5/2.54,9.5/2.54])
for name, u_mean, midl, shape, size in zip(names, U, midlat, shapes, sizes):
    beta = 2 * omega * np.cos(np.radians(midl)) / R  # [1/ms]
    lengh_x = 2 * np.pi * R * np.sin(np.radians(midl))  # [m]
    lengh_y = 2 * np.pi * R * 43 / 360  # [m]
    c_list = []
    for wn in wavenumbers:
        lambda_x = lengh_x/wn
        lambda_y = lengh_y/0.5
        c = u_mean - beta/((2*np.pi/lambda_x)**2 + (2*np.pi/lambda_y)**2)
        #c = u_mean - beta/((2*np.pi/lambda_x)**2)
        c_list.append(c)
    plt.scatter(wavenumbers, c_list, marker = shape, c = 'dimgray', label = name, s = size)
plt.legend(loc='best')
plt.yticks(fontsize=11)
plt.xticks(wavenumbers, fontsize=11)
plt.axhline(0, 0, 7, color='steelblue', linestyle='--')
plt.ylabel('Phase speed [m/s]', fontsize=12)
plt.xlabel('Wavenumber', fontsize=12)
plt.savefig(path_figures + f'Phase_speed_all_barotropic_average_l.jpg', dpi=500)
plt.savefig(path_figures + f'Phase_speed_all_barotropic_average_l.svg', dpi=500)
#plt.show()
plt.close()



aaaa
names = names[:-1]
U = U[:-1]
midlat = midlat[:-1]
colors =colors[:-1]

colors = ['darkslateblue', 'mediumpurple', 'skyblue', 'silver']
shapes = ['o', 'x', 'D', '*']
fig = plt.figure(figsize=[7,5])
for name, u_mean, midl, shape, color in zip(names, U, midlat, shapes, colors):
    beta = 2 * omega * np.cos(np.radians(midl)) / R  # [1/ms]
    lengh_x = 2 * np.pi * R * np.sin(np.radians(midl))  # [m]
    lengh_y = 2 * np.pi * R * 43 / 360  # [m]
    c_list = []
    for wn in wavenumbers:
        lambda_x = lengh_x/wn
        lambda_y = lengh_y/0.5
        c = u_mean - beta/((2*np.pi/lambda_x)**2 + (2*np.pi/lambda_y)**2)
        #c = u_mean - beta/((2*np.pi/lambda_x)**2)
        c_list.append(c)
    plt.scatter(wavenumbers, c_list, marker = shape, c = color, label = name, s = 32)
plt.legend(loc='best')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(wavenumbers, fontsize=12)
plt.axhline(0, 0, 7, color='teal', linestyle='--')
plt.ylabel('Phase speed [m/s]', fontsize=12)
plt.xlabel('Wavenumber', fontsize=12)
plt.savefig(path_figures + f'Phase_speed_all_barotropic_average_l_noCAM.jpg', dpi=500)
#plt.show()
plt.close()