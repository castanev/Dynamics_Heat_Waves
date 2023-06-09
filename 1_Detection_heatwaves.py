# This code is for the detection of heatwaves. It's based on the methodology used by Teng, 2013
# Here, a heat wave event is defined as at least 5 consecutive days following:
# i). More than 5% of the domain (US) has daily averaged SAT exceeding the threshold value
# ii). Centre of these warm points does not move faster than 5◦ latitude or longitude per day

# Threshold: 97.5 percentile for historical t within a 15-day window centred on the day (for each day and each gridpoint)
# To avoid contamination, we use only events that have no heat wave days in the preceding 20 day

# NOTE: 
# For the detection of heat waves in the Dry Core GCM, the threshold is defined as the 97.5 percentile of all the data (the model has no seasons)

from Functions import *
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.colors
import os

import matplotlib.pyplot as plt
import cartopy
from cartopy import crs
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
from scipy import ndimage

# INPUTS TO CHANGE ===========================================================================================================================
# name: It can be NCEP or the name of the experiment (output of the model)
# nc_name: name of the file containing the surface air temperature. 't.atmos_daily.nc' for the dry core GCM or 't1000_NCEP.nc' for NCEP
# var: the variable name in the nc file. 't1000' for NCEP, 'temp' for the dry core GCM 
# min_duration: minimun consecutive days for a heat wave event
# vel: vel is a vector to limit the speed of the center of the warm points [max. degrees, number of days]
# threshold_value: percentile 
# lat_minHW, lat_maxHW, lon_minHW, lon_minHW: to define the area where the events will be detected
# delete_days is the number of days to cut (only for output models). 

<<<<<<< HEAD
name = 'NCEP'  
nc_name = 't2m_NCEP.nc'
var = 't1000'
vel = [5, 1]
=======
# name = 'NCEP'  
# nc_name = 't2m_NCEP.nc'
# var = 't1000'
# vel = [5, 1]
# min_duration = 5
# threshold_value = 97.5
# lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat=45 #235 to 290
# seasons = True 
# topography = True
# delete_days = ''
# path_data = f'/scratch/brown/castanev/{name}/'

# name = 'CAM'  
# nc_name = f't1000_{name}.nc'
# var = 't1000'
# vel = [5, 1]
# min_duration = 5
# threshold_value = 97.5
# lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat=45 #235 to 290
# seasons = True 
# topography = True
# delete_days = 1000
# path_data = f'/scratch/brown/castanev/{name}/'

name = 'exp1_Held_Suarez'  
nc_name = f't.atmos_daily.nc'
var = 'temp'
vel = [7, 1]
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
min_duration = 5
min_area = 900000 # [km2]
threshold_value = 97.5
<<<<<<< HEAD
lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat=45 #235 to 290
seasons = True 
topography = True
delete_days = ''
path_data = f'/scratch/brown/castanev/{name}/'

# name = 'CAM'  
# nc_name = f't1000_{name}.nc'
# var = 't1000'
# vel = [5, 1]
# min_duration = 5
# min_area = 900000 # [km2]
# threshold_value = 97.5
# lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat=45 #235 to 290
# seasons = True 
# topography = True
# delete_days = 1000
# path_data = f'/scratch/brown/castanev/{name}/'


# name = 'exp1_Held_Suarez'  
# nc_name = f't.atmos_daily.nc'
# var = 'temp'
# vel = [7, 1]
# min_duration = 5
# min_area = 900000 # [km2]
# threshold_value = 97.5
# #lat_minHW = 25; lat_maxHW = 50; lon_minHW = 245; lon_maxHW = 275; midlat=45  #exp1, exp2
# lat_minHW = 25; lat_maxHW = 50; lon_minHW = 240; lon_maxHW = 270; midlat=45   #exp3
# seasons = False 
# topography = False
# delete_days = 1000
# path_data = f'/scratch/brown/castanev/DryCore_Wu/output/{name}/post_processed/output/'
=======
lat_minHW = 30; lat_maxHW = 50; lon_minHW = 245; lon_maxHW = 275; midlat=45
seasons = False 
topography = False
delete_days = 1000
path_data = f'/scratch/brown/castanev/DryCore_Wu/output/{name}/post_processed/output/'
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e


# ============================================================================================================================================
# path_data = f'{os.path.abspath(os.getcwd())}/{name}/Data/'
# path_figures = f'{os.path.abspath(os.getcwd())}/{name}/Figures/'
# path_outputs = f'{os.path.abspath(os.getcwd())}/{name}/'

path_figures = f'/home/castanev/Heat-waves-dynamics/{name}/Figures/'
path_outputs = f'/home/castanev/Heat-waves-dynamics/{name}/'

methodology = 'Teng'
vel_str = f'vel{str(vel[0])}{str(vel[1])}'

ncfile = Dataset(f'{path_data}{nc_name}')
time = np.array(ncfile['time'])
t_k = np.array(ncfile[var])  #(t, lat, lon) °K
t = pd.DataFrame(data=np.reshape(t_k, [t_k.shape[0], t_k.shape[1] * t_k.shape[2]]))
t = t - 273.15 # °C
t = t.values.reshape(t_k.shape[0], t_k.shape[1], t_k.shape[2])
lats = np.array(ncfile['lat'])
lons = np.array(ncfile['lon'])
<<<<<<< HEAD
=======
time = np.array(ncfile['time'])
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e

# Spatial cut: United States
pos_lats = np.where((lats >= lat_minHW) & (lats <= lat_maxHW))
pos_lons = np.where((lons >= lon_minHW) & (lons <= lon_maxHW))
lats_US = lats[pos_lats]
lons_US = lons[pos_lons]
total_areal = (lat_maxHW-lat_minHW)*111*(lon_maxHW-lon_minHW)*111
percentage_min_area = min_area/total_areal
min_grid_5 = round(lats_US.shape[0] * lons_US.shape[0] * percentage_min_area)
t_US = t[:, pos_lats[0], :]
t_US = t_US[:, :, pos_lons[0]]

print(f"Total de días analizados = {t_US.shape[0]}")
<<<<<<< HEAD
print(f'% of the total grids corresponds to = {percentage_min_area}')  
=======
print(f'5% of the total grids corresponds to = {min_grid_5}')  
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e

if seasons == True:
    if   name == 'NCEP': dates_d = np.array([dt.datetime(1800,1,1) + dt.timedelta(hours = int(time[i])) for i in range(len(time))])
    elif name in ['CAM', 'LENS']:  
<<<<<<< HEAD
        timei = [str(int(time[i]))[-4:] for i in range(len(time))]
        time = []
        for i in range(147000):
            x = dt.datetime.strptime(timei[0] + '1700', '%m%d%Y') + i * dt.timedelta(days=1)    
            if x.month == 2 and x.day == 29: continue
            time.append(x)
        
        dates_d = np.array(time[delete_days:])
        t_US = t_US[delete_days:]
        t_US = t_US[:len(dates_d)]
        
=======
        time = [str(int(time[i]))[-4:] for i in range(len(time))]
        dates_d = np.array([dt.datetime.strptime(i, '%m%d') for i in time])
        t_US = t_US[delete_days:,:,:]
        dates_d = dates_d[delete_days:]
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e

    Month = np.array([ii.month for ii in dates_d])
    df_t = pd.DataFrame(index=dates_d, data=np.reshape(t_US, [t_US.shape[0], t_US.shape[1] * t_US.shape[2]]))

    days_summer = np.array([dt.datetime(2021, 6, 1) + relativedelta(days=int(xx)) for xx in range(92)])  # random year

    threshold = np.zeros([len(days_summer), len(lats_US), len(lons_US)])
    for i, d in enumerate(days_summer):
        t_pos1 = df_t.index.get_indexer_for((df_t.loc[(pd.DatetimeIndex(df_t.index).month == d.month) & (pd.DatetimeIndex(df_t.index).day == d.day)].index))
        #Positions of the 15-day window centred on the day of the year of the potential heat wave day:
        #t_pos = [np.concatenate((t_pos1, t_pos1 + iii)) for iii in range(-7, 8)]
        # t_pos = [np.concatenate((t_pos1, t_pos1 + iii)) for iii in range(-10, 21)]
        # t_pos = np.unique(t_pos)
        # t_pos = t_pos[t_pos < t_US.shape[0]] # For the last days of the dataset
        data = t_US[t_pos1, :, :]
        threshold_i = np.percentile(data, threshold_value, axis=0)
        threshold[i, :, :] = threshold_i


    df_threshold = pd.DataFrame(index=days_summer,
                                data=np.reshape(threshold, [days_summer.shape[0], t_US.shape[1] * t_US.shape[2]]))

    pos_summer = np.where([ii in [6, 7, 8] for ii in Month])[0]
    dates_summer = dates_d[pos_summer]
    t_summer = t_US[pos_summer, :, :]

    pos_heat_wavesi = []
    for i, date in enumerate(dates_summer):

        threshold_pos = df_threshold.index.get_indexer_for(
            (df_threshold.loc[(df_threshold.index.month == date.month) & (df_threshold.index.day == date.day)].index))
        
        if i+1 == dates_summer.shape[0]: break
        date_2 = dates_summer[i+1]
        threshold_pos_2 = df_threshold.index.get_indexer_for(
            (df_threshold.loc[(df_threshold.index.month == date_2.month) & (df_threshold.index.day == date_2.day)].index))
        
        # Condition i). More than 5% of the domain (US) has daily averaged SAT exceeding the threshold value
        cond_1 = t_US[pos_summer[i], :, :] > threshold[threshold_pos, :, :]
        cond_1_2 = t_US[pos_summer[i] + vel[1], :, :] > threshold[threshold_pos_2, :, :]
        grid_cont = np.count_nonzero(cond_1)

        # Condition ii): Centre of these warm points does not move faster than 5◦ latitude or longitude per day
        if pos_summer[i] + vel[1] == t_US.shape[0]: break

        pos_max = ndimage.measurements.center_of_mass(cond_1[0])
        pos_max2 = ndimage.measurements.center_of_mass(cond_1_2[0])
<<<<<<< HEAD

        dif_lats = np.abs(pos_max2[0] - pos_max[0]) * np.abs(lats_US[1]-lats_US[0])
        dif_lons = np.abs(pos_max2[1] - pos_max[1]) * np.abs(lons_US[1]-lons_US[0])
=======

        dif_lats = np.abs(pos_max2[0] - pos_max[0]) * np.abs(lats_US[1]-lats_US[0])
        dif_lons = np.abs(pos_max2[1] - pos_max[1]) * np.abs(lons_US[1]-lons_US[0])

        if (grid_cont > min_grid_5) and (dif_lats < vel[0]) and (dif_lons < vel[0]):
            pos_heat_wavesi.append(pos_summer[i])  # dates_d[pos_summer[i]] is the date that meets both conditions
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e

        if (grid_cont > min_grid_5) and (dif_lats < vel[0]) and (dif_lons < vel[0]):
            pos_heat_wavesi.append(pos_summer[i])  # dates_d[pos_summer[i]] is the date that meets both conditions

elif seasons == False:
    t_US = t_US[delete_days:,:,:]
    df_t = pd.DataFrame(data=np.reshape(t_US, [t_US.shape[0], t_US.shape[1] * t_US.shape[2]]))
    threshold = np.percentile(t_US, threshold_value, axis=0)

    pos_heat_wavesi = []
    for pos in range(t_US.shape[0]):
        if pos + vel[1] == t_US.shape[0]: break
        # Condition i). More than 5% of the domain (US) has daily averaged SAT exceeding the threshold value
        cond_1 = t_US[pos, :, :] > threshold
        cond_1_2 = t_US[pos + vel[1], :, :] > threshold
        grid_cont = np.count_nonzero(cond_1)

        # Condition ii): Centre of these warm points does not move faster than 5◦ latitude or longitude per day
        # Center defined as the point with the max temperature in the domain     
        pos_max = ndimage.measurements.center_of_mass(cond_1)
        pos_max2 = ndimage.measurements.center_of_mass(cond_1_2)
<<<<<<< HEAD
=======

        dif_lats = np.abs(pos_max2[0] - pos_max[0]) * np.abs(lats_US[1]-lats_US[0])
        dif_lons = np.abs(pos_max2[1] - pos_max[1]) * np.abs(lons_US[1]-lons_US[0])

        if (grid_cont > min_grid_5) and (dif_lats < vel[0]) and (dif_lons < vel[0]):
            pos_heat_wavesi.append(pos)  # dates_d[pos_summer[i]] is the date that meets both conditions
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e

        dif_lats = np.abs(pos_max2[0] - pos_max[0]) * np.abs(lats_US[1]-lats_US[0])
        dif_lons = np.abs(pos_max2[1] - pos_max[1]) * np.abs(lons_US[1]-lons_US[0])

<<<<<<< HEAD
        if (grid_cont > min_grid_5) and (dif_lats < vel[0]) and (dif_lons < vel[0]):
            pos_heat_wavesi.append(pos)  

=======
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
duration_hw, pos_day1_hw = duration_heat_waves(pos_heat_wavesi, min_duration)
print(f'Number of heat waves events: {len(duration_hw)}')

# Probability distribution function for duration
bins = np.arange(np.unique(duration_hw)[0], np.unique(duration_hw)[-1] + 1, 1)
Hist, bins1 = np.histogram(duration_hw, len(bins))
PDF_temp = Hist / len(duration_hw)


fig = plt.figure(figsize=[4, 4])
df = pd.DataFrame({'x': bins, 'PDF': PDF_temp})
df.plot.bar(x='x', y='PDF', rot=0, color='dimgray', width=.4)
plt.ylabel('PDF', fontsize=13)
plt.xlabel('Duration (d)', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, labelcolor='linecolor')
#plt.show()
plt.savefig(path_figures + f'SAT_PDF_{methodology}_{vel_str}.png', dpi=500)
plt.close()


pos_heat_waves = []
for day1, dur in zip(pos_day1_hw, duration_hw): 
    pos_heat_waves.append([day1 + i for i in range(dur)])
pos_heat_waves = [item for sublist in pos_heat_waves for item in sublist]
print(f'Number heat waves days: {len(pos_heat_waves)}')


# Saving the resume of statistics in a .csv
resume = pd.read_csv(f'{path_outputs}/../resume_heatWaves_statistics.csv', index_col = 0)
resume.loc[name, f'HW days'] = len(pos_heat_waves)
resume.loc[name, f'HW events'] = len(duration_hw)
if seasons == True: resume.loc[name, f'Analized days'] = len(pos_summer)
else: resume.loc[name, f'Analized days'] = t_US.shape[0]
resume.to_csv(f'{path_outputs}/../resume_heatWaves_statistics.csv')
print(len(pos_heat_waves))
# Saving the position of HW days in a .csv
pos_heat_waves_serie = pd.Series(pos_heat_waves)
pos_heat_waves_serie.to_csv(f'{path_outputs}/resume_positions_HWdays_{name}_{methodology}.csv')
<<<<<<< HEAD


# Saving PDF of duration in a .csv
pos_heat_waves_serie = pd.Series(index = bins[:10], data = PDF_temp[:10])
pos_heat_waves_serie.to_csv(f'{path_outputs}/PDF_duration_{name}_{methodology}.csv')
=======
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e

t_US_hw = t_US[pos_heat_waves, :, :]


# Calculating anomalies
if seasons == True: anom_t_US = anomalies_seasons(df_t)
elif seasons == False: anom_t_US = anomalies_noseasons(df_t)

anom_t_US = anom_t_US.values.reshape(anom_t_US.shape[0], lats_US.shape[0], lons_US.shape[0])
anom_t_US_hw = anom_t_US[pos_heat_waves, :, :]


# # Creating list with all heat waves events 
# intensities = []
# for dur_i, pos1_i in zip(duration_hw, pos_day1_hw):
#     intensity_i = round(np.max(np.mean(anom_t_US[pos1_i:pos1_i+dur_i-1], axis=0)),2)
#     intensities.append(intensity_i)
# pos_heat_waves_serie = pd.DataFrame(data = {'Date 0': dates_d[pos_day1_hw], 'Duration': duration_hw, 'Intensity': intensities}, index = pos_day1_hw)
# pos_heat_waves_serie.to_csv(f'{path_outputs}/Heat_waves_events_list.csv')


# Ploting intensity
RdYlBu_v2_list = ['rgb(224,243,248)','rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)','rgb(49,54,149)']
my_cmap = matplotlib.colors.ListedColormap(RdYlBu_v2_list, name='RdYlBu')

<<<<<<< HEAD

if name in ['LENS', 'NCEP']:
    li, ls, intervalos, limite, color = 0.5, 2.5, 15, 1, 'RdYlBu_r'
    bounds = np.round(np.linspace(li, ls, intervalos), 3)
    colormap = center_white_anom(color, intervalos, bounds, limite)
    maps2(lons_US, np.ndarray.round(lats_US,2), 0.7, 2.5 + np.abs(bounds[0] - bounds[1]), np.nanmean(anom_t_US_hw, axis=0), r'SAT anomalies [°C]',
        colormap, path_figures + f'SAT_mean_anomt_hw_{methodology}_{vel_str}.png', topography)


    # For the Figure to explain the methodology - Threshold before, during and after an event
    pos_event_pre_after = [26822, 26824, 26826, 26828, 26830, 26832, 26834, 26836, 26838]
    fig = plt.figure(figsize=[10, 7])
    for i, pos in enumerate(pos_event_pre_after):
        date = dates_d[pos]
        ax = fig.add_subplot(3, 3, 1+i, projection=crs.PlateCarree(central_longitude=180))
        ax.add_feature(cartopy.feature.BORDERS, lw=0.5)

        ax.outline_patch.set_edgecolor('None')
        ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

        im2 = ax.contourf(lons_US, lats_US, np.nanmean(threshold[threshold_pos,:,:], axis=0), cmap='Reds', extend='both', \
                        levels=np.arange(17, 35, 0.2), transform=crs.PlateCarree())
        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                        linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        gl.ylocator = mticker.FixedLocator([30, 45])
        gl.xlocator = mticker.FixedLocator([-120, -100, -80])
        gl.xlabel_style = {'size': 9, 'color': 'dimgrey'}
        gl.ylabel_style = {'size': 9, 'color': 'dimgrey'}
        ax.set_title(f'SAT {date.strftime("%Y-%m-%d")}', fontsize=9, color='dimgrey')
    cbaxes = fig.add_axes([0.31, 0.07, 0.38, 0.018])
    cb = plt.colorbar(im2, orientation="horizontal", pad=0.13, cax=cbaxes, format='%.0f')
    tick_locator = ticker.MaxNLocator(nbins=7)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.set_label('SAT [°C]', fontsize=10, color='dimgrey')
    cb.outline.set_edgecolor(None)
    cb.ax.tick_params(labelcolor='dimgrey', color='dimgrey', labelsize=9)
    #plt.show()
    plt.savefig(f'{path_figures}/Methodology/Methodology_Threshold_{methodology}_{vel_str}.png', dpi=200)
    plt.close()


    # For the Figure to explain the methodology - SAT before, during and after an event
    fig = plt.figure(figsize=[10, 7])
    for i, pos in enumerate(pos_event_pre_after):
        date = dates_d[pos]
        ax = fig.add_subplot(3, 3, 1+i, projection=crs.PlateCarree(central_longitude=180))
        ax.add_feature(cartopy.feature.BORDERS, lw=0.5)

        ax.outline_patch.set_edgecolor('None')
        ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

        im2 = ax.contourf(lons_US, lats_US, t_US[pos,:,:], cmap='Reds', extend='both', \
                        levels=np.arange(17, 35, 0.2), transform=crs.PlateCarree())
        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                        linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        gl.ylocator = mticker.FixedLocator([30, 45])
        gl.xlocator = mticker.FixedLocator([-120, -100, -80])
        gl.xlabel_style = {'size': 9, 'color': 'dimgrey'}
        gl.ylabel_style = {'size': 9, 'color': 'dimgrey'}
        ax.set_title(f'SAT {date.strftime("%Y-%m-%d")}', fontsize=9, color='dimgrey')
    cbaxes = fig.add_axes([0.31, 0.07, 0.38, 0.018])
    cb = plt.colorbar(im2, orientation="horizontal", pad=0.13, cax=cbaxes, format='%.0f')
    tick_locator = ticker.MaxNLocator(nbins=7)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.set_label('SAT [°C]', fontsize=10, color='dimgrey')
    cb.outline.set_edgecolor(None)
    cb.ax.tick_params(labelcolor='dimgrey', color='dimgrey', labelsize=9)
    #plt.show()
    plt.savefig(f'{path_figures}/Methodology/Methodology_SAT_{methodology}_{vel_str}.png', dpi=200)
    plt.close()

    fig = plt.figure(figsize=[10, 7])
    #plt.title('Difference SAT - Threshold', fontsize=11, color='dimgrey')
    #fig.suptitle('Difference SAT - Threshold', fontsize=11, color='dimgrey', va='baseline')
    plt.axis('off')
    for i, pos in enumerate(pos_event_pre_after):
        date = dates_d[pos]
        threshold_pos = df_threshold.index.get_indexer_for(
            (df_threshold.loc[(df_threshold.index.month == date.month) & (df_threshold.index.day == date.day)].index))
        dif = t_US[pos,:,:] - np.nanmean(threshold[threshold_pos,:,:], axis=0)
        cond_1 = t_US[pos, :, :] > np.nanmean(threshold[threshold_pos,:,:], axis=0)
        grid_cont = np.count_nonzero(cond_1)
        percent_cond = round(grid_cont*100/(lats_US.shape[0] * lons_US.shape[0]), 1)

        ax = fig.add_subplot(3, 3, i+1, projection=crs.PlateCarree(central_longitude=180))
        ax.add_feature(cartopy.feature.BORDERS, lw=0.5)

        ax.outline_patch.set_edgecolor('None')
        ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

        li, ls, intervalos, limite, color = 0, 3, 15, 0.1, 'BuPu'
        bounds = np.round(np.linspace(li, ls, intervalos), 3)
        colormap = center_white_anom(color, intervalos, bounds, limite)

        im3 = ax.contourf(lons_US, lats_US, dif, cmap=colormap, extend='both', \
                        levels=np.arange(0, 3, 0.01), transform=crs.PlateCarree())

        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                        linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        gl.ylocator = mticker.FixedLocator([30, 45])
        gl.xlocator = mticker.FixedLocator([-120, -100, -80])
        gl.xlabel_style = {'size': 9, 'color': 'dimgrey'}
        gl.ylabel_style = {'size': 9, 'color': 'dimgrey'}
        ax.set_title(f'{date.strftime("%Y-%m-%d")} \n {percent_cond}%', fontsize=9, color='dimgrey')
    cbaxes = fig.add_axes([0.31, 0.07, 0.38, 0.018])
    cb = plt.colorbar(im3, orientation="horizontal", pad=0.13, cax=cbaxes, format='%.1f')
    tick_locator = ticker.MaxNLocator(nbins=7)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.set_label('SAT - Threshold [°C]', fontsize=10, color='dimgrey')
    cb.outline.set_edgecolor(None)
    cb.ax.tick_params(labelcolor='dimgrey', color='dimgrey', labelsize=9)
    #plt.show()
    plt.savefig(f'{path_figures}/Methodology/Methodology_Difference_{methodology}_{vel_str}.png', dpi=200)
    plt.close()


    # dates_serie = dates_d[pos_event_pre_after[0]:pos_event_pre_after[-1]]
    # pos_max = ndimage.measurements.center_of_mass(t_US[pos_event_pre_after[3]])
    # fig = plt.figure(figsize=(15, 7), dpi=150)
    # ax = fig.add_subplot(1, 1, 1)
    # threshold_pos1 = df_threshold.index.get_indexer_for(
    #     (df_threshold.loc[(df_threshold.index.month == dates_serie[0].month) & (df_threshold.index.day == dates_serie[0].day)].index))
    # threshold_pos2 = df_threshold.index.get_indexer_for(
    #     (df_threshold.loc[(df_threshold.index.month == dates_serie[-1].month) & (df_threshold.index.day == dates_serie[-1].day)].index))
    # print(threshold_pos1)
    # plt.plot(dates_serie, t_US[pos_event_pre_after[0]:pos_event_pre_after[-1],int(pos_max[0]), int(pos_max[1])],color = 'b', ls = '--', lw = 2, label = 'T')
    # plt.plot(dates_serie, threshold[threshold_pos1[0]:threshold_pos2[0]+1,int(pos_max[0]), int(pos_max[1])], color = 'tomato', ls = '--', lw = 1.6, label = 'Threshold')
    # plt.xlabel('Forecasted date', fontsize=14)
    # plt.ylabel('T [K]', fontsize=14)
    # plt.yticks(fontsize=13)
    # plt.xticks(fontsize=13)
    # plt.legend(fontsize=12)
    # plt.savefig(f'{path_figures}/Methodology/Serie{dates_serie[0].strftime("%Y-%m-%d")}.png', dpi=500) 
    # plt.close()



elif name in ['exp1_Held_Suarez', 'exp3_NCEPasymm_noSeason_noTop', 'exp2_NCEPsymm_noSeason_noTop']:
    li, ls, intervalos, limite, color = 0.7, 3, 15, 0.25, 'RdYlBu_r'
    bounds = np.round(np.linspace(li, ls, intervalos), 3)
    colormap = center_white_anom(color, intervalos, bounds, limite)
    maps2(lons_US, np.ndarray.round(lats_US,2), 0.7, 3 + np.abs(bounds[0] - bounds[1]), np.nanmean(anom_t_US_hw, axis=0), r'SAT anomalies [°C]',
        colormap, path_figures + f'SAT_mean_anomt_hw_{methodology}_{vel_str}.png', topography)

=======
li, ls, intervalos, limite, color = 0.7, 3, 15, 1.4, 'RdYlBu_r'
bounds = np.round(np.linspace(li, ls, intervalos), 3)
colormap = center_white_anom(color, intervalos, bounds, limite)
maps2(lons_US, np.ndarray.round(lats_US,2), 0.7, 3 + np.abs(bounds[0] - bounds[1]), np.nanmean(anom_t_US_hw, axis=0), r'SAT anomalies [°C]',
      colormap, path_figures + f'SAT_mean_anomt_hw_{methodology}_{vel_str}.png', topography)
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e







