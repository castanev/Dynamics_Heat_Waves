# This code is to compare the statistics between reanalysis data and the experiments of the model 
import csv 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

path_outputs = f'/home/castanev/Heat-waves-dynamics/Comparison/'

statistics = pd.read_csv(f'{path_outputs}/../resume_heatWaves_statistics.csv')

names = statistics.iloc[:,0]
short_names = ['NCEP','CTR','ZOB','ROB']
HW_days = statistics.loc[:,'HW days']
HW_events = statistics.loc[:,'HW events']
Analized_days = statistics.loc[:,'Analized days']

#fig = plt.figure(figsize=[6.8,6.5])
#fig = plt.figure(figsize=[9.5/2.54,11.5/2.54])
fig = plt.figure(figsize=[11.5/2.54,11.5/2.54])
#plt.scatter(short_names, HW_days/Analized_days, s = 30, marker = 'o', c = 'dimgray')
plt.bar(short_names, HW_days/Analized_days, width = 0.16, color = 'dimgray')
plt.hlines(statistics.loc[0,'HW days']/statistics.loc[0,'Analized days'], 0, 3, ls = '--', colors = 'steelblue')
plt.ylim(0,0.04)
plt.xticks(fontsize =12)
plt.yticks(fontsize = 12)
plt.ylabel('Heatwaves days / Analyzed days', fontsize = 12)
plt.savefig(path_outputs + f'Frequency.png', dpi=500)
plt.savefig(path_outputs + f'Frequency.svg')
plt.savefig(path_outputs + f'Frequency.eps')
plt.close()

markers = ['', 'o', 'x', 'd']
#fig = plt.figure(figsize=[6.5,5])
fig = plt.figure(figsize=[11.5/2.54,10.5/2.54])
for name, marker, short_name in zip(names, markers, short_names): 
    durations_PDF = pd.read_csv(f'{path_outputs}../{name}/PDF_duration_{name}_Teng.csv', index_col = 0)
    if name == 'NCEP': plt.plot(np.arange(5,13,1), durations_PDF[:8], ls = '--', c = 'steelblue',label = short_name)
    else: plt.plot(np.arange(5,5+durations_PDF.shape[0],1)[:8], durations_PDF[:8], ls = '-', c = 'dimgray', lw = 0.8, marker = marker, label = short_name)
plt.ylim(0,0.6)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('Duration', fontsize = 12)
plt.ylabel('PDF', fontsize = 12)
plt.xticks(fontsize=12)
plt.savefig(path_outputs + f'Durations_PDF.png', dpi=500)
plt.savefig(path_outputs + f'Durations_PDF.svg')
plt.savefig(path_outputs + f'Durations_PDF.eps')
plt.close() 
    
aaa

HW_days.iloc[1:] = np.nan
fig = plt.figure(figsize=[6.8,6.5])
#plt.scatter(short_names, HW_days/Analized_days, s = 30, marker = 'o', c = 'dimgray')
plt.bar(short_names, HW_days/Analized_days, width = 0.16, color = 'dimgray')
plt.hlines(statistics.loc[0,'HW days']/statistics.loc[0,'Analized days'], 0, 3, ls = '--', colors = 'steelblue')
plt.ylim(0,0.04)
plt.xticks(rotation=15, ha='right', fontsize =13)
plt.yticks(fontsize = 13)
plt.ylabel('Heatwaves days / Analyzed days', fontsize = 13)
plt.savefig(path_outputs + f'Frequency_NCEP.png', dpi=200)
plt.close()

markers = ['', 'o', 'x', 'd']
plt.figure(figsize=[6.5,5])
for name, marker, short_name in zip(names, markers, short_names): 
    durations_PDF = pd.read_csv(f'{path_outputs}../{name}/PDF_duration_{name}_Teng.csv', index_col = 0)
    if name == 'NCEP': plt.plot(np.arange(5,13,1), durations_PDF[:8], ls = '--', c = 'steelblue',label = short_name)
    else: continue
plt.ylim(0,0.6)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('Duration', fontsize = 12)
plt.ylabel('PDF', fontsize = 12)
plt.xticks(fontsize=12)
plt.savefig(path_outputs + f'Durations_PDF_NCEP.png', dpi=200)
plt.close() 
    

from PIL import Image, ImageDraw

path_gif_figures = '/home/castanev/Heat-waves-dynamics/NCEP/Figures/Composites_gif/'
filenames = np.sort(os.listdir(f'{path_gif_figures}'))
files = np.arange(0,len(filenames), 1).astype('str')
images = []
for filename in files:
    images.append(Image.open(F'{path_gif_figures}{filename}.png'))
images[0].save('/home/castanev/Heat-waves-dynamics/NCEP/Figures/Composites.gif', save_all=True, append_images=images[1:], optimize=False, duration=250, loop=0)

path_gif_figures = '/home/castanev/Heat-waves-dynamics/exp2_NCEPsymm_noSeason_noTop/Figures/Composites_gif/'
filenames = np.sort(os.listdir(f'{path_gif_figures}'))
files = np.arange(0,len(filenames), 1).astype('str')
images = []
for filename in files:
    images.append(Image.open(F'{path_gif_figures}{filename}.png'))
images[0].save('/home/castanev/Heat-waves-dynamics/exp2_NCEPsymm_noSeason_noTop/Figures/Composites.gif', save_all=True, append_images=images[1:], optimize=False, duration=250, loop=0)

path_gif_figures = '/home/castanev/Heat-waves-dynamics/exp3_NCEPasymm_noSeason_noTop/Figures/Composites_gif/'
filenames = np.sort(os.listdir(f'{path_gif_figures}'))
files = np.arange(0,len(filenames), 1).astype('str')
images = []
for filename in files:
    images.append(Image.open(F'{path_gif_figures}{filename}.png'))
images[0].save('/home/castanev/Heat-waves-dynamics/exp3_NCEPasymm_noSeason_noTop/Figures/Composites.gif', save_all=True, append_images=images[1:], optimize=False, duration=250, loop=0)