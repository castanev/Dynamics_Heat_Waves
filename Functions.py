# import pandas as pd
# import datetime as dt
# import scipy as scp
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy as scp
# from cartopy import crs
# import cartopy
# import matplotlib.ticker as mticker
# import matplotlib.colors as mcolors
# import os

# ================================= PLOTS ===============================================
def center_white_anom(cmap, num, bounds, limite):
    import matplotlib as mpl
    import numpy as np
    import matplotlib.cm 
    barcmap = matplotlib.cm.get_cmap(cmap, num)
    barcmap.set_bad(color='white', alpha=0.5)
    bar_vals = barcmap(np.arange(num))  # extract those values as an array
    pos = np.arange(num)
    centro = pos[(bounds >= -limite) & (bounds <= limite)]
    for i in centro:
        bar_vals[i] = [1, 1, 1, 1]  # change the middle value
    newcmap = mpl.colors.LinearSegmentedColormap.from_list("new" + cmap, bar_vals)
    return newcmap


def maps1(x, y, minn, maxx, matriz,  cmap, path, norm, units='', topography = ''):
    import matplotlib.pyplot as plt
    import cartopy
    from cartopy import crs
    import numpy as np
    import matplotlib.ticker as mticker

<<<<<<< HEAD
    fig = plt.figure(figsize=[6, 5.5])
=======
    fig = plt.figure(figsize=[6.5, 5.5])
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree(central_longitude=180))
    
    if topography == True:
        ax.add_feature(cartopy.feature.BORDERS, lw=0.5)
        ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

<<<<<<< HEAD
    im = ax.contourf(x, y[:], matriz[:,:], levels=np.arange(minn, maxx, 4), cmap = cmap, norm=norm, extend='both', transform=crs.PlateCarree())
    #r = ax.contour(x, y, matriz, levels=15, colors='k', linewidths=0.5)
    
    ax.set_yticks([-60,-25,0,25,60])
    ax.tick_params(axis='both', labelsize=12)
=======
    im = ax.contourf(x, y[:], matriz[:,:], levels=np.arange(minn, maxx, 1), cmap = cmap, norm=norm, extend='both', transform=crs.PlateCarree())
    #r = ax.contour(x, y, matriz, levels=15, colors='k', linewidths=0.5)
    
    ax.set_yticks([-60,-25,0,25,60])
    ax.tick_params(axis='both', labelsize=10)
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
    gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                      linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
<<<<<<< HEAD
    gl.left_labels = False
    gl.ylocator = mticker.FixedLocator([-60,-25,0,25,60])
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    
    cb = plt.colorbar(im, orientation="horizontal", pad=0.1, format='%.1f', shrink=0.8)
    cb.set_label(units, fontsize=11, color='dimgrey')
    cb.ax.tick_params(labelsize=12)
    plt.savefig(path, dpi=500)
=======
    gl.ylocator = mticker.FixedLocator([-60,-25,0,25,60])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    cb = plt.colorbar(im, orientation="horizontal", pad=0.1, format='%.1f', shrink=0.8)
    cb.set_label(units, fontsize=11, color='dimgrey')
    cb.ax.tick_params(labelsize=10)
    plt.savefig(path, dpi=200)
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
    plt.close()

def maps_vert(y, z, minn, maxx, matriz,  cmap, path, norm, units=''):
    import matplotlib.pyplot as plt
    import cartopy
    from cartopy import crs
    import numpy as np
    import matplotlib.ticker as mticker

    fig = plt.figure(figsize=[6.5, 5.5])
    ax = fig.add_subplot(1, 1, 1)

    im = ax.contourf(y, z[:], matriz[:,:], levels=np.arange(minn, maxx, 1), cmap = cmap, norm=norm, extend='both')
    #r = ax.contour(x, y, matriz, levels=15, colors='k', linewidths=0.5)
    plt.gca().invert_yaxis()
    ax.tick_params(axis='both', labelsize=10)
    
    cb = plt.colorbar(im, orientation="horizontal", pad=0.1, format='%.1f', shrink=0.8)
    cb.set_label(units, fontsize=11, color='dimgrey')
    cb.ax.tick_params(labelsize=10)
    plt.savefig(path, dpi=500)
    plt.close()


def maps_vert(y, z, minn, maxx, matriz,  cmap, path, norm, units=''):
    import matplotlib.pyplot as plt
    import cartopy
    from cartopy import crs
    import numpy as np
    import matplotlib.ticker as mticker

    fig = plt.figure(figsize=[6.5, 4.7])
    ax = fig.add_subplot(1, 1, 1)

    im = ax.contourf(y, z[:], matriz[:,:], levels=np.arange(minn, maxx, 1), cmap = cmap, norm=norm, extend='both')
    #r = ax.contour(x, y, matriz, levels=15, colors='k', linewidths=0.5)
    plt.gca().invert_yaxis()
    ax.tick_params(axis='both', labelsize=12)
    
    cb = plt.colorbar(im, orientation="horizontal", pad=0.1, format='%.1f', shrink=0.8)
    cb.set_label(units, fontsize=12, color='dimgrey')
    cb.ax.tick_params(labelsize=12)
    plt.savefig(path, dpi=500)
    plt.close()


def maps2(Lons, Lats, minn, maxx, matriz, var, cmap, path, topography):
    import matplotlib.pyplot as plt
    import cartopy
    import numpy as np
    import matplotlib.ticker as mticker
    from cartopy import crs
    fig = plt.figure(figsize=[6.5, 5.5])
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree(central_longitude=180))
    
    if topography == True:
        ax.add_feature(cartopy.feature.BORDERS, lw=0.5)
        ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

    im = ax.contourf(Lons, Lats, matriz, cmap=cmap, extend='both', \
<<<<<<< HEAD
                     levels=np.arange(minn, maxx, 0.2), transform=crs.PlateCarree())
=======
                     levels=np.arange(minn, maxx, 0.5), transform=crs.PlateCarree())
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
    #ax.set_yticks([30, 35, 40, 45])
    #ax.tick_params(axis='both', labelsize=14)

    gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                      linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([30, 35, 40, 45])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    cbaxes = fig.add_axes([0.2, 0.12, 0.6, 0.030])
    cb = plt.colorbar(im, orientation="horizontal", pad=0.1, cax=cbaxes, format='%.1f')
    cb.set_label(var, fontsize=10, color='dimgrey')
    cb.ax.tick_params(labelsize=9)
<<<<<<< HEAD
    plt.savefig(path, dpi=500)
=======
    plt.savefig(path, dpi=200)
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
    plt.close()


def composites_coastlines(lats, lons, Matriz_t, min_t, max_t, Matriz_sf, levels_sf, Matriz_env, levels_env, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, cmap, path, var = '', topography = ''):
    import matplotlib.pyplot as plt
    import cartopy
    import numpy as np
    import matplotlib.ticker as mticker
    from cartopy import crs
<<<<<<< HEAD
    #fig = plt.figure(figsize=[17, 18])
    fig = plt.figure(figsize=[19/2.54, 23/2.54])
=======
    fig = plt.figure(figsize=[15, 15])
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
    for i, tit in enumerate(['Day -20', 'Day -15', 'Day -10', 'Day -5', 'Day 0', 'Day 5']):
        ax = fig.add_subplot(6, 1, i + 1, projection=crs.PlateCarree(central_longitude=180))
        if topography == True:
            ax.add_feature(cartopy.feature.COASTLINE, lw=0.6, zorder=11)

        ax.set_title(tit, fontsize = 15)
<<<<<<< HEAD
        im = ax.contourf(lons, lats, Matriz_t[i, :, :], cmap=cmap, extend='both', levels=np.linspace(min_t, max_t, 15),transform=crs.PlateCarree())
=======
        im = ax.contourf(lons, lats, Matriz_t[i, :, :], cmap=cmap, extend='both', levels=np.arange(min_t, max_t, 0.2),transform=crs.PlateCarree())
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
        im2 = ax.contour(lons, lats, Matriz_env[i, :, :], extend='both', levels=levels_env, colors='k', linewidths=2.3,transform=crs.PlateCarree())
        ax.clabel(im2, inline=True, fontsize=10, fmt='%1.1f')
        levels_sf = np.array(levels_sf)
        im3 = ax.contour(lons, lats, Matriz_sf[i, :, :], extend='both', levels=levels_sf[levels_sf>0], colors='firebrick', linewidths=2.1, transform=crs.PlateCarree())
        im4 = ax.contour(lons, lats, Matriz_sf[i, :, :], extend='both', levels=levels_sf[levels_sf<0], colors='mediumblue', negative_linestyles = 'dashed', linewidths=2,transform=crs.PlateCarree())

        ax.plot([lon_minHW, lon_maxHW], [lat_minHW, lat_minHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_minHW, lon_maxHW], [lat_maxHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_minHW, lon_minHW], [lat_minHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_maxHW, lon_maxHW], [lat_minHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.set_yticks([30, 50])
        ax.tick_params(axis='both', labelsize=14)
        plt.ylim(20,70)

        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.ylocator = mticker.FixedLocator([30, 50])
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

    cbaxes = fig.add_axes([0.3, 0.06, 0.4, 0.015])
    cb = plt.colorbar(im, orientation="horizontal", pad=0.2, cax=cbaxes, format='%.1f')
    cb.set_label(var, fontsize=15)
    cb.outline.set_edgecolor('k')
    cb.ax.tick_params(labelsize=14)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.92,
                    wspace=0.1,
<<<<<<< HEAD
                    hspace=0.2)
=======
                    hspace=0.1)
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e

    plt.savefig(path, dpi=500)
    plt.close()

<<<<<<< HEAD
def composites_gif(lats, lons, Matriz_t, min_t, max_t, Matriz_sf, levels_sf, Matriz_env, levels_env, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, cmap, path, var = '', topography = ''):
    import matplotlib.pyplot as plt
    import cartopy
    import numpy as np
    import matplotlib.ticker as mticker
    from cartopy import crs
    for i in range(25):
        tit = f'Day {i-20}'
        fig = plt.figure(figsize=[15,4.2])
        ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree(central_longitude=180))
        if topography == True:
            ax.add_feature(cartopy.feature.COASTLINE, lw=0.6, zorder=11)

        ax.set_title(tit, fontsize = 15)
        im = ax.contourf(lons, lats, Matriz_t[i, :, :], cmap=cmap, extend='both', levels=np.linspace(min_t, max_t, 15),transform=crs.PlateCarree())
        im2 = ax.contour(lons, lats, Matriz_env[i, :, :], extend='both', levels=levels_env, colors='k', linewidths=2.3,transform=crs.PlateCarree())
        ax.clabel(im2, inline=True, fontsize=10, fmt='%1.1f')
        levels_sf = np.array(levels_sf)
        im3 = ax.contour(lons, lats, Matriz_sf[i, :, :], extend='both', levels=levels_sf[levels_sf>0], colors='firebrick', linewidths=2.1, transform=crs.PlateCarree())
        im4 = ax.contour(lons, lats, Matriz_sf[i, :, :], extend='both', levels=levels_sf[levels_sf<0], colors='mediumblue', negative_linestyles = 'dashed', linewidths=2,transform=crs.PlateCarree())

        ax.plot([lon_minHW, lon_maxHW], [lat_minHW, lat_minHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_minHW, lon_maxHW], [lat_maxHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_minHW, lon_minHW], [lat_minHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_maxHW, lon_maxHW], [lat_minHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.set_yticks([30, 50])
        ax.tick_params(axis='both', labelsize=14)
        plt.ylim(20,70)

        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.ylocator = mticker.FixedLocator([30, 50])
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

        cbaxes = fig.add_axes([0.3, 0.13, 0.4, 0.045])
        cb = plt.colorbar(im, orientation="horizontal", pad=0.2, cax=cbaxes, format='%.1f')
        cb.set_label(var, fontsize=12)
        cb.outline.set_edgecolor('k')
        cb.ax.tick_params(labelsize=12)
        plt.subplots_adjust(left=0.1,
                        bottom=0.2,
                        right=0.9,
                        top=0.98,
                        wspace=0.04,
                        hspace=0.08)

        plt.savefig(f'{path}{i}.png', dpi=500)
        plt.close()
    
=======
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
def composites_coastlines_2(lats, lons, Matriz_t, min_t, max_t, Matriz_sf, levels_sf, Matriz_env, levels_env, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, cmap, path, var, topography):
    import matplotlib.pyplot as plt
    import cartopy
    import numpy as np
    import matplotlib.ticker as mticker
    from cartopy import crs
    fig = plt.figure(figsize=[15, 17])
    for i, tit in enumerate(['Day -11', 'Day -9', 'Day -6', 'Day -3', 'Day 0', 'Day 5']):
        ax = fig.add_subplot(6, 1, i + 1, projection=crs.PlateCarree(central_longitude=180))
        if topography == True:
            ax.add_feature(cartopy.feature.COASTLINE, lw=0.6, zorder=11)

        ax.set_title(tit, fontsize = 17)
        im = ax.contourf(lons, lats, Matriz_t[i, :, :], cmap=cmap, extend='both', levels=np.arange(min_t, max_t, 0.2),transform=crs.PlateCarree())
        im2 = ax.contour(lons, lats, Matriz_env[i, :, :], extend='both', levels=levels_env, colors='k', linewidths=2.5,transform=crs.PlateCarree())
        ax.clabel(im2, inline=True, fontsize=14, fmt='%1.1f')
        levels_sf = np.array(levels_sf)
        im3 = ax.contour(lons, lats, Matriz_sf[i, :, :], extend='both', levels=levels_sf[levels_sf>0], colors='firebrick', linewidths=2.3, transform=crs.PlateCarree())
        im4 = ax.contour(lons, lats, Matriz_sf[i, :, :], extend='both', levels=levels_sf[levels_sf<0], colors='mediumblue', negative_linestyles = 'dashed', linewidths=2.1,transform=crs.PlateCarree())

        ax.plot([lon_minHW, lon_maxHW], [lat_minHW, lat_minHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_minHW, lon_maxHW], [lat_maxHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_minHW, lon_minHW], [lat_minHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_maxHW, lon_maxHW], [lat_minHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.set_yticks([30, 50])
        ax.tick_params(axis='both', labelsize=14)
        plt.ylim(20,70)

        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.ylocator = mticker.FixedLocator([30, 50])
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

    cbaxes = fig.add_axes([0.3, 0.06, 0.4, 0.015])
    cb = plt.colorbar(im, orientation="horizontal", pad=0.2, cax=cbaxes, format='%.1f')
    cb.set_label(var, fontsize=15)
    cb.outline.set_edgecolor('k')
    cb.ax.tick_params(labelsize=14)
    plt.subplots_adjust(left=0.04,
                    bottom=0.1,
                    right=0.96,
                    top=0.94,
                    wspace=0.12,
                    hspace=0.12)
<<<<<<< HEAD

    plt.savefig(path, dpi=500)
    plt.close()


def composites_coastlines_event_2(lats, lons, Matriz_t, min_t, max_t, Matriz_z500, levels_z500, Matriz_z1000, levels_z1000, lat_minHW_local, lat_maxHW_local, lon_minHW_local, lon_maxHW_local, cmap, path, var, topography):
    import matplotlib.pyplot as plt
    import cartopy
    import numpy as np
    import matplotlib.ticker as mticker
    from cartopy import crs
    fig = plt.figure(figsize=[7, 12])
    for i, tit in enumerate(['Day -12', 'Day -8', 'Day -4', 'Day 0', 'Day 5']):
        ax = fig.add_subplot(6, 1, i + 1, projection=crs.PlateCarree(central_longitude=180))
        if topography == True:
            ax.add_feature(cartopy.feature.COASTLINE, lw=0.6, zorder=11)

        ax.set_title(tit, fontsize = 17)
        im = ax.contourf(lons, lats, Matriz_t[i, :, :], cmap=cmap, extend='both', levels=np.arange(min_t, max_t, 0.2),transform=crs.PlateCarree())
        im3 = ax.contour(lons, lats, Matriz_z1000[i, :, :], extend='both', levels=levels_z1000, colors='yellowgreen', linewidths=2.3, transform=crs.PlateCarree())
        im2 = ax.contour(lons, lats, Matriz_z500[i, :, :], extend='both', levels=levels_z500, colors='k', linewidths=2.5,transform=crs.PlateCarree())
        ax.clabel(im2, inline=True, fontsize=11, fmt='%1.0f')
        
        if tit == 'Day 0':
            ax.plot([lon_minHW_local, lon_maxHW_local], [lat_minHW_local, lat_minHW_local], transform=crs.PlateCarree(), color='b', lw=2)
            ax.plot([lon_minHW_local, lon_maxHW_local], [lat_maxHW_local, lat_maxHW_local], transform=crs.PlateCarree(), color='b', lw=2)
            ax.plot([lon_minHW_local, lon_minHW_local], [lat_minHW_local, lat_maxHW_local], transform=crs.PlateCarree(), color='b', lw=2)
            ax.plot([lon_maxHW_local, lon_maxHW_local], [lat_minHW_local, lat_maxHW_local], transform=crs.PlateCarree(), color='b', lw=2)
       
        ax.set_yticks([30,45])
        ax.tick_params(axis='both', labelsize=14)


        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.ylocator = mticker.FixedLocator([30, 45])
        gl.ylocator = mticker.FixedLocator([-120, -100, -80])
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

    cbaxes = fig.add_axes([0.2, 0.06, 0.6, 0.015])
    cb = plt.colorbar(im, orientation="horizontal", pad=0.2, cax=cbaxes, format='%.1f')
    cb.set_label(var, fontsize=15)
    cb.outline.set_edgecolor('k')
    cb.ax.tick_params(labelsize=14)
    plt.subplots_adjust(left=0.007,
                    bottom=0.007,
                    right=0.993,
                    top=0.94,
                    wspace=0.1,
                    hspace=0.45)

    plt.savefig(path, dpi=500)
    plt.close()


def magnitude(vector):
    import math
    return math.sqrt(sum(pow(element, 2) for element in vector))
 
# def composites_coastlines_event_barbs(lats, lons, Matriz_t, min_t, max_t, Matriz_z500, levels_z500, Matriz_u, Matriz_v, lat_minHW_local, lat_maxHW_local, lon_minHW_local, lon_maxHW_local, cmap, path, var, topography):
#     import matplotlib.pyplot as plt
#     import cartopy
#     import numpy as np
#     import matplotlib.ticker as mticker
#     from cartopy import crs
#     import math
#     fig = plt.figure(figsize=[7, 14])
#     for i, tit in enumerate(['Day -4', 'Day 0', 'Day 4']):
#         ax = fig.add_subplot(3, 1, 1 + i, projection=crs.PlateCarree())
#         if topography == True:
#             ax.add_feature(cartopy.feature.COASTLINE, lw=0.6, zorder=11)

#         ax.set_title(tit, fontsize = 17)
#         im = ax.contourf(lons, lats, Matriz_t[i, :, :], cmap=cmap, extend='both', levels=np.arange(min_t, max_t, 0.2),transform=crs.PlateCarree())
#         im2 = ax.contour(lons, lats, Matriz_z500[i, :, :], extend='both', levels=levels_z500, colors='k', linewidths=2.5,transform=crs.PlateCarree())
#         ax.clabel(im2, inline=True, fontsize=11, fmt='%1.0f')
#         x, y = np.meshgrid(lons, lats)
#         im3 = plt.quiver(x, y, Matriz_u[i, :, :], Matriz_v[i, :, :], angles='xy', scale_units='xy', scale=10)
#         #V_magnitude_max = magnitude([abs(Matriz_u).max(), abs(Matriz_v).max()])
#         #plt.quiverkey(im3, 0.95, 1.02, V_magnitude_max, label=f'{round(V_magnitude_max,0)} [m/s]')

#         if tit == 'Day 0':
#             ax.plot([lon_minHW_local, lon_maxHW_local], [lat_minHW_local, lat_minHW_local], transform=crs.PlateCarree(), color='b', lw=2)
#             ax.plot([lon_minHW_local, lon_maxHW_local], [lat_maxHW_local, lat_maxHW_local], transform=crs.PlateCarree(), color='b', lw=2)
#             ax.plot([lon_minHW_local, lon_minHW_local], [lat_minHW_local, lat_maxHW_local], transform=crs.PlateCarree(), color='b', lw=2)
#             ax.plot([lon_maxHW_local, lon_maxHW_local], [lat_minHW_local, lat_maxHW_local], transform=crs.PlateCarree(), color='b', lw=2)
    
#         ax.set_yticks([30,45])
#         ax.tick_params(axis='both', labelsize=14)

#         gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
#         gl.top_labels = False
#         gl.right_labels = False
#         gl.ylocator = mticker.FixedLocator([30, 45])
#         gl.ylocator = mticker.FixedLocator([-120, -100, -80])
#         gl.xlabel_style = {'size': 14}
#         gl.ylabel_style = {'size': 14}

#     cbaxes = fig.add_axes([0.2, 0.06, 0.6, 0.015])
#     cb = plt.colorbar(im, orientation="horizontal", pad=0.2, cax=cbaxes, format='%.1f')
#     cb.set_label(var, fontsize=15)
#     cb.outline.set_edgecolor('k')
#     cb.ax.tick_params(labelsize=14)
#     plt.subplots_adjust(left=0.007,
#                     bottom=0.007,
#                     right=0.993,
#                     top=0.94,
#                     wspace=0.1,
#                     hspace=0.3)

#     plt.savefig(path, dpi=500)
#     plt.close()



def map_event_barbs(lats, lons, Matriz_t, min_t, max_t, Matriz_z500, levels_z500, Matriz_u, Matriz_v, lat_minHW_local, lat_maxHW_local, lon_minHW_local, lon_maxHW_local, cmap, path, var, topography):
    import matplotlib.pyplot as plt
    import cartopy
    import numpy as np
    import matplotlib.ticker as mticker
    from cartopy import crs
    import math
    fig = plt.figure(figsize=[8, 5.5])
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())
    if topography == True:
        ax.add_feature(cartopy.feature.COASTLINE, lw=0.6, zorder=11)

    im = ax.contourf(lons, lats, Matriz_t, cmap=cmap, extend='both', levels=np.arange(min_t, max_t, 0.2),transform=crs.PlateCarree())
    im2 = ax.contour(lons, lats, Matriz_z500, extend='both', levels=levels_z500, colors='mediumblue', linewidths=1.9,transform=crs.PlateCarree())
    ax.clabel(im2, inline=True, fontsize=11, fmt='%1.0f')
    x, y = np.meshgrid(lons, lats)
    im3 = plt.quiver(x, y, Matriz_u, Matriz_v, angles='xy', scale_units='xy', scale=5)
    V_magnitude_max = magnitude([abs(Matriz_u).max(), abs(Matriz_v).max()])
    plt.quiverkey(im3, 0.95, 1.02, V_magnitude_max, label=f'{round(V_magnitude_max,0)} [m/s]')

    ax.set_yticks([30,45])
    ax.tick_params(axis='both', labelsize=14)

    gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([30, 45])
    gl.ylocator = mticker.FixedLocator([-120, -100, -80])
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    cbaxes = fig.add_axes([0.2, 0.06, 0.6, 0.015])
    cb = plt.colorbar(im, orientation="horizontal", pad=0.2, cax=cbaxes, format='%.1f')
    cb.set_label(var, fontsize=15)
    cb.outline.set_edgecolor('k')
    cb.ax.tick_params(labelsize=14)
    # plt.subplots_adjust(left=0.05,
    #                 bottom=0.03,
    #                 right=0.95,
    #                 top=0.97,
    #                 wspace=0.1,
    #                 hspace=0.3)

    plt.savefig(path, dpi=500)
    plt.close()
=======
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e

    plt.savefig(path, dpi=500)
    plt.close()
    
def hovmoller(time_lags, lons, Matriz_t, cmap_t, norm_t, min_t, max_t, Matriz_sf, cmap_sf, norm_sf, min_sf, max_sf, Matriz_env, levels_env, path, var_1 = '', var_2 = ''):
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import numpy as np
    RdYlBu_list = ['rgb(165,0,38)','rgb(215,48,39)','rgb(244,109,67)','rgb(253,174,97)','rgb(254,224,144)','rgb(255,255,191)','rgb(224,243,248)','rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)','rgb(49,54,149)']
    my_cmap = matplotlib.colors.ListedColormap(RdYlBu_list, name='RdYlBu')

    fig = plt.figure(figsize=[7,4])
    ax = fig.add_subplot(1, 2, 1)
    im = ax.contourf(lons, time_lags, Matriz_t[:,:], cmap=cmap_t, extend='both', levels=np.linspace(min_t, max_t, 15))
    plt.ylim(-15,15)
    plt.ylabel(r'Time lag', fontsize=8)
    plt.xlabel(r'Relative longitude', fontsize=8)
    ax.set_yticks([-15,-10,-5,0,5,10,15])
    ax.set_xticks([-180,-90,0,90,180])
    ax.tick_params(labelsize=7)
    ax.axhline(0, ls = '--', color='dimgray', lw = 0.3)
    ax.axvline(0, ls = '--', color='dimgray', lw = 0.3)
    cbaxes = fig.add_axes([0.11, 0.12, 0.35, 0.03])
    #cb = plt.colorbar(cm.ScalarMappable(norm=norm_t, cmap=cmap_t), orientation="horizontal", pad=0.2, cax=cbaxes, format='%.2f')
    cb = plt.colorbar(im, orientation="horizontal", pad=0.2, cax=cbaxes, format='%.1f')
    cb.set_label(var_1, fontsize=7, color='dimgrey')
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(labelcolor='dimgrey', color='dimgrey', labelsize=7)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.contour(lons, time_lags, Matriz_env[:, :], extend='both', levels=levels_env, colors='k', linewidths=0.9)
    ax2.clabel(im2, inline=True, fontsize=6, fmt='%1.1f')
    im3 = ax2.contourf(lons, time_lags, Matriz_sf[:, :],  cmap=cmap_sf, extend='both', levels=np.linspace(min_sf, max_sf, 15))
    plt.ylim(-15,15)
    plt.xlabel(r'Relative longitude', fontsize=8)
    ax2.set_yticks([-15,-10,-5,0,5,10,15])
    ax2.set_xticks([-180,-90,0,90,180])
    ax2.tick_params(labelsize=7)
    ax2.axhline(0, ls = '--', color='dimgray', lw = 0.3)
    ax2.axvline(0, ls = '--', color='dimgray', lw = 0.3)
    cbaxes = fig.add_axes([0.55, 0.12, 0.35, 0.03])
    #cb = plt.colorbar(cm.ScalarMappable(norm=norm_sf, cmap=cmap_sf), orientation="horizontal", pad=0.2, cax=cbaxes, format='%.2f')
    cb = plt.colorbar(im3, orientation="horizontal", pad=0.2, cax=cbaxes, format='%1.2f')
    cb.set_label(var_2, fontsize=7, color='dimgrey')
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(labelcolor='dimgrey', color='dimgrey', labelsize=7)

    plt.subplots_adjust(left=0.08,
                    bottom=0.3,
                    right=0.92,
                    top=0.92,
                    wspace=0.15,
                    hspace=0.2)
    plt.savefig(path, dpi=500)
    plt.close()


# ================================= DETECTION OF HEAT WAVES ===============================================

# To calculate the duration and the position of the first day of each event
def duration_heat_waves(pos_hw, min_duration):
    count = 1
    duration_hw = []
    pos_day1_hw = []
    for i in range(len(pos_hw[:-1])):
        if pos_hw[i] + 1 == pos_hw[i + 1]:
            count += 1
        else:
            print(count)
            if count >= min_duration and len(pos_day1_hw) == 0:
                duration_hw.append(count)
                pos_day1_hw.append(pos_hw[i - count + 1])
            elif count >= min_duration and abs((pos_day1_hw[-1] + duration_hw[-1]) - pos_hw[i - count + 1]) > 20:
                duration_hw.append(count)
                pos_day1_hw.append(pos_hw[i - count + 1])
            count = 1
    return duration_hw, pos_day1_hw


def anomalies_seasons(df_VAR):
    import numpy as np
    import pandas as pd
    ANOMA = df_VAR * np.nan
    for i in np.arange(1, 13):
        mes = df_VAR.loc[pd.DatetimeIndex(df_VAR.index).month == i]
        temp = mes * np.nan
        Nodays = pd.DatetimeIndex(mes.index).day.max()
        if np.isnan(Nodays) == True: continue
        for j in np.arange(1, Nodays + 1):
            dia = mes.loc[pd.DatetimeIndex(mes.index).day == j]
            media = dia.mean()
            anoma = dia - media
            temp[pd.DatetimeIndex(temp.index).day == j] = anoma
        ANOMA.loc[pd.DatetimeIndex(temp.index)] = temp
<<<<<<< HEAD
    return ANOMA


def climatology_threshold_series(df_VAR, percentile):
    import numpy as np
    import pandas as pd
    import datetime as dt
    from dateutil.relativedelta import relativedelta
    indexx = [dt.datetime(2012, 1, 1) + relativedelta(days=int(xx)) for xx in range(366)]
    CLIMA = pd.Series(index = indexx)
    THRESHOLD = pd.Series(index = indexx)
    df_VAR_threshold = round(df_VAR.rolling(15, center=True).mean(),2)
    for i in np.arange(1, 13):
        mes = df_VAR.loc[pd.DatetimeIndex(df_VAR.index).month == i]
        mes_threshold = df_VAR_threshold.loc[pd.DatetimeIndex(df_VAR.index).month == i]
        temp = CLIMA.loc[pd.DatetimeIndex(CLIMA.index).month == i]
        temp_threshold = THRESHOLD.loc[pd.DatetimeIndex(CLIMA.index).month == i]
        Nodays = pd.DatetimeIndex(mes.index).day.max()
        if np.isnan(Nodays) == True: continue
        for j in np.arange(1, Nodays + 1):
            dia = mes.loc[pd.DatetimeIndex(mes.index).day == j]
            media = dia.mean()
            temp.loc[pd.DatetimeIndex(temp.index).day == j] = media
            dia_threshold = mes_threshold.loc[pd.DatetimeIndex(mes_threshold.index).day == j]
            threshold = np.nanpercentile(dia_threshold, percentile)
            temp_threshold[pd.DatetimeIndex(temp_threshold.index).day == j] = threshold
        CLIMA.loc[pd.DatetimeIndex(temp.index)] = temp
        THRESHOLD.loc[pd.DatetimeIndex(temp_threshold.index)] = temp_threshold
    return CLIMA, THRESHOLD


def anomalies_seasons_movil(df_VAR):
    import numpy as np
    import pandas as pd
    ANOMA = df_VAR * np.nan
    mm_31 = round(df_VAR.rolling(5, center=True).mean(),2)
    for i in np.arange(1, 13):
        mes = mm_31.loc[pd.DatetimeIndex(df_VAR.index).month == i]
        temp = mes * np.nan
        Nodays = pd.DatetimeIndex(mes.index).day.max()
        if np.isnan(Nodays) == True: continue
        for j in np.arange(1, Nodays + 1):
            dia = mes.loc[pd.DatetimeIndex(mes.index).day == j]
            media = round(dia.mean(),2)
            anoma = dia - media
            temp[pd.DatetimeIndex(temp.index).day == j] = anoma
        ANOMA.loc[pd.DatetimeIndex(temp.index)] = temp
    return ANOMA

def anomalies_seasons_movil1(df_VAR):
    import numpy as np
    import pandas as pd
    ANOMA = df_VAR * np.nan
    mm_31 = round(df_VAR.rolling(21, center=True).mean(),2)
    for i in np.arange(1, 13):
        mes_mm = mm_31.loc[pd.DatetimeIndex(df_VAR.index).month == i]
        mes = df_VAR.loc[pd.DatetimeIndex(df_VAR.index).month == i]
        temp = mes * np.nan
        Nodays = pd.DatetimeIndex(mes.index).day.max()
        if np.isnan(Nodays) == True: continue
        for j in np.arange(1, Nodays + 1):
            dia_mm = mes_mm.loc[pd.DatetimeIndex(mes_mm.index).day == j]
            dia = mes.loc[pd.DatetimeIndex(mes.index).day == j]
            media = round(dia_mm.mean(),2)
            anoma = dia - media
            temp[pd.DatetimeIndex(temp.index).day == j] = anoma
        ANOMA.loc[pd.DatetimeIndex(temp.index)] = temp
=======
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
    return ANOMA


def anomalies_seasons_movil2(df_VAR):
    import numpy as np
    import pandas as pd
    ANOMA = df_VAR * np.nan
    mm_31 = round(df_VAR.rolling(21, center=True).mean(),2)
    mm_3 = round(df_VAR.rolling(5, center=True).mean(),2)
    for i in np.arange(1, 13):
        mes_mm = mm_31.loc[pd.DatetimeIndex(mm_31.index).month == i]
        mes = mm_3.loc[pd.DatetimeIndex(mm_3.index).month == i]
        temp = mes * np.nan
        Nodays = pd.DatetimeIndex(mes.index).day.max()
        if np.isnan(Nodays) == True: continue
        for j in np.arange(1, Nodays + 1):
            dia_mm = mes_mm.loc[pd.DatetimeIndex(mes_mm.index).day == j]
            dia = mes.loc[pd.DatetimeIndex(mes.index).day == j]
            media = round(dia_mm.mean(),2)
            anoma = dia - media
            temp[pd.DatetimeIndex(temp.index).day == j] = anoma
        ANOMA.loc[pd.DatetimeIndex(temp.index)] = temp
    return ANOMA

def anomalies_noseasons(df_VAR):
    import numpy as np
    ANOMA = df_VAR * np.nan
    media = df_VAR.mean()
    for i in range(df_VAR.shape[0]):
        dia = df_VAR.loc[i]
        anoma = dia - media
        ANOMA.iloc[i] = anoma
    return ANOMA

#anomalies_seasons_model(df_t, days, Month)


# ================================= EVOLUTION OF HEAT WAVES ===============================================
def subseasonal_anomalies(df_VAR):
    import numpy as np
    import pandas as pd
    years = np.unique(np.array([ii.year for ii in df_VAR.index]))
    ANOMA = df_VAR * np.nan
    for i in years:
        pos_y = np.where(df_VAR.index.year == i)[0]
        year = df_VAR.iloc[pos_y]
        temp = year * np.nan
        for j in ([12,1,2], [3,4,5], [6,7,8], [9,10,11]):
            pos_season = np.where((pd.to_datetime(year.index).month == j[0]) ^ (pd.to_datetime(year.index).month == j[1]) ^ (pd.to_datetime(year.index).month == j[2]))[0]
            season = year.iloc[pos_season]
            media = season.mean()
            anoma = season - media
            temp.iloc[pos_season] = anoma
        ANOMA.iloc[pos_y] = temp
    return ANOMA



def hilbert_filtered(x, k_min, k_max, N=None, axis=-1, path=''):
    from scipy import linalg, fft as sp_fft
    import numpy as np
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``np.imag(hilbert(x))``, and the
    original signal from ``np.real(hilbert(x))``.

    Examples
    --------
    In this example we use the Hilbert transform to determine the amplitude
    envelope and instantaneous frequency of an amplitude-modulated signal.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import hilbert, chirp

    >>> duration = 1.0
    >>> fs = 400.0
    >>> samples = int(fs*duration)
    >>> t = np.arange(samples) / fs

    We create a chirp of which the frequency increases from 20 Hz to 100 Hz and
    apply an amplitude modulation.

    >>> signal = chirp(t, 20.0, t[-1], 100.0)
    >>> signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

    The amplitude envelope is given by magnitude of the analytic signal. The
    instantaneous frequency can be obtained by differentiating the
    instantaneous phase in respect to time. The instantaneous phase corresponds
    to the phase angle of the analytic signal.

    >>> analytic_signal = hilbert(signal)
    >>> amplitude_envelope = np.abs(analytic_signal)
    >>> instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    >>> instantaneous_frequency = (np.diff(instantaneous_phase) /
    ...                            (2.0*np.pi) * fs)

    >>> fig, (ax0, ax1) = plt.subplots(nrows=2)
    >>> ax0.plot(t, signal, label='signal')
    >>> ax0.plot(t, amplitude_envelope, label='envelope')
    >>> ax0.set_xlabel("time in seconds")
    >>> ax0.legend()
    >>> ax1.plot(t[1:], instantaneous_frequency)
    >>> ax1.set_xlabel("time in seconds")
    >>> ax1.set_ylim(0.0, 120.0)
    >>> fig.tight_layout()

    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal
    .. [2] Leon Cohen, "Time-Frequency Analysis", 1995. Chapter 2.
    .. [3] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal
           Processing, Third Edition, 2009. Chapter 12.
           ISBN 13: 978-1292-02572-8

    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = sp_fft.fft(x, N, axis=axis)
    pow = np.abs(Xf/len(x))**2
    freq = np.fft.fftfreq(len(x), 1)

    # fig = plt.figure(figsize = [10,4])
    # plt.plot(freq*360, pow*100/np.sum(pow),'o-', color='dimgray', markersize=1.8)
    # plt.xscale('log')
    # plt.legend(loc="upper right", frameon=True)
    # plt.ylabel('[%] Potencia')
    # plt.xlabel('Frecuencia')
    # plt.title('Espectro  de Fourier')
    # #plt.show()
    # #plt.savefig(path, dpi=200) 
    # plt.close()

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    
    #filter = np.where((np.abs(freq*360) < k_min) ^ (np.abs(freq*360) > k_max))[0]
    wavelengths = (1/freq)*np.abs(x[1]-x[0])*111.321 #[km]
    filter = np.where((wavelengths < 2800) ^ (wavelengths > 10000))[0] # Wavenumbers 3 to 11
    Xf_filtered = np.copy(Xf)
    Xf_filtered[filter] = 0

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = sp_fft.ifft(Xf_filtered * h, axis=axis)
    return x


<<<<<<< HEAD
# def hilbert_filtered(x, k_min, k_max, N=None, axis=-1, path=''):
#     from scipy import linalg, fft as sp_fft
#     import numpy as np
#     """
#     Compute the analytic signal, using the Hilbert transform.
#     The transformation is done along the last axis by default.
#     Parameters
#     ----------
#     x : array_like
#         Signal data.  Must be real.
#     N : int, optional
#         Number of Fourier components.  Default: ``x.shape[axis]``
#     axis : int, optional
#         Axis along which to do the transformation.  Default: -1.
#     Returns
#     -------
#     xa : ndarray
#         Analytic signal of `x`, of each 1-D array along `axis`
#     Notes
#     -----
#     The analytic signal ``x_a(t)`` of signal ``x(t)`` is:
#     .. math:: x_a = F^{-1}(F(x) 2U) = x + i y
#     where `F` is the Fourier transform, `U` the unit step function,
#     and `y` the Hilbert transform of `x`. [1]_
#     In other words, the negative half of the frequency spectrum is zeroed
#     out, turning the real-valued signal into a complex signal.  The Hilbert
#     transformed signal can be obtained from ``np.imag(hilbert(x))``, and the
#     original signal from ``np.real(hilbert(x))``.
#     Examples
#     --------
#     In this example we use the Hilbert transform to determine the amplitude
#     envelope and instantaneous frequency of an amplitude-modulated signal.
#     >>> import numpy as np
#     >>> import matplotlib.pyplot as plt
#     >>> from scipy.signal import hilbert, chirp
#     >>> duration = 1.0
#     >>> fs = 400.0
#     >>> samples = int(fs*duration)
#     >>> t = np.arange(samples) / fs
#     We create a chirp of which the frequency increases from 20 Hz to 100 Hz and
#     apply an amplitude modulation.
#     >>> signal = chirp(t, 20.0, t[-1], 100.0)
#     >>> signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
#     The amplitude envelope is given by magnitude of the analytic signal. The
#     instantaneous frequency can be obtained by differentiating the
#     instantaneous phase in respect to time. The instantaneous phase corresponds
#     to the phase angle of the analytic signal.
#     >>> analytic_signal = hilbert(signal)
#     >>> amplitude_envelope = np.abs(analytic_signal)
#     >>> instantaneous_phase = np.unwrap(np.angle(analytic_signal))
#     >>> instantaneous_frequency = (np.diff(instantaneous_phase) /
#     ...                            (2.0*np.pi) * fs)
#     >>> fig, (ax0, ax1) = plt.subplots(nrows=2)
#     >>> ax0.plot(t, signal, label='signal')
#     >>> ax0.plot(t, amplitude_envelope, label='envelope')
#     >>> ax0.set_xlabel("time in seconds")
#     >>> ax0.legend()
#     >>> ax1.plot(t[1:], instantaneous_frequency)
#     >>> ax1.set_xlabel("time in seconds")
#     >>> ax1.set_ylim(0.0, 120.0)
#     >>> fig.tight_layout()
#     References
#     ----------
#     .. [1] Wikipedia, "Analytic signal".
#            https://en.wikipedia.org/wiki/Analytic_signal
#     .. [2] Leon Cohen, "Time-Frequency Analysis", 1995. Chapter 2.
#     .. [3] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal
#            Processing, Third Edition, 2009. Chapter 12.
#            ISBN 13: 978-1292-02572-8
#     """
#     x = np.asarray(x)
#     if np.iscomplexobj(x):
#         raise ValueError("x must be real.")
#     if N is None:
#         N = x.shape[axis]
#     if N <= 0:
#         raise ValueError("N must be positive.")

#     Xf = sp_fft.fft(x, N, axis=axis)
#     pow = np.abs(Xf/len(x))**2
#     freq = np.fft.fftfreq(len(x), 1)

#     # fig = plt.figure(figsize = [10,4])
#     # plt.plot(freq*360, pow*100/np.sum(pow),'o-', color='dimgray', markersize=1.8)
#     # plt.xscale('log')
#     # plt.legend(loc="upper right", frameon=True)
#     # plt.ylabel('[%] Potencia')
#     # plt.xlabel('Frecuencia')
#     # plt.title('Espectro  de Fourier')
#     # #plt.show()
#     # #plt.savefig(path, dpi=200) 
#     # plt.close()

#     h = np.zeros(N)
#     if N % 2 == 0:
#         h[0] = h[N // 2] = 1
#         h[1:N // 2] = 2
#     else:
#         h[0] = 1
#         h[1:(N + 1) // 2] = 2

#     filter = np.where((np.abs(freq*360) < k_min) ^ (np.abs(freq*360) > k_max))[0]
#     Xf_filtered = np.copy(Xf)
#     Xf_filtered[filter] = 0

#     if x.ndim > 1:
#         ind = [np.newaxis] * x.ndim
#         ind[axis] = slice(None)
#         h = h[tuple(ind)]
#     x = sp_fft.ifft(Xf_filtered * h, axis=axis)
#     return x

=======
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
def calculate_composites_event(pos_HW, matriz):
    import numpy as np
    dic_composites = {}

    time_lags = np.arange(-20, 21, 1)
    for num in time_lags:
        dic_composites[num] = []  
        dic_composites[num].append(pos_HW+num)  

    composites_matrix_complete =  np.zeros((len(time_lags), matriz.shape[1], matriz.shape[2]))
    for ii, lag in enumerate(np.sort(list(dic_composites))):
        composites_matrix_complete[ii] = np.mean(matriz[dic_composites[lag]], axis = 0)

    day_0 = np.mean(matriz[dic_composites[0]], axis=0)
    day_5 = np.mean(matriz[dic_composites[-5]], axis=0)
    day_10 = np.mean(matriz[dic_composites[-10]], axis=0)
    day_15 = np.mean(matriz[dic_composites[-15]], axis=0)
    day_20 = np.mean(matriz[dic_composites[-20]], axis=0)
    day_5_post = np.mean(matriz[dic_composites[5]], axis=0)

    composites_matrix = np.zeros((6, day_0.shape[0], day_0.shape[1]))
    for i, matriz_i in enumerate([day_20, day_15, day_10, day_5, day_0, day_5_post]):
        composites_matrix[i,:,:] = matriz_i
    
    return composites_matrix, composites_matrix_complete


<<<<<<< HEAD
def calculate_composites_event_2(pos_HW, matriz):
    import numpy as np
    dic_composites = {}

    time_lags = np.arange(-20, 21, 1)
    for num in time_lags:
        dic_composites[num] = []  
        dic_composites[num].append(pos_HW+num)  

    composites_matrix_complete =  np.zeros((len(time_lags), matriz.shape[1], matriz.shape[2]))
    for ii, lag in enumerate(np.sort(list(dic_composites))):
        composites_matrix_complete[ii] = np.mean(matriz[dic_composites[lag]], axis = 0)

    day_0 = np.mean(matriz[dic_composites[0]], axis=0)
    day_4 = np.mean(matriz[dic_composites[-4]], axis=0)
    day_8 = np.mean(matriz[dic_composites[-8]], axis=0)
    day_12 = np.mean(matriz[dic_composites[-12]], axis=0)
    day_5_post = np.mean(matriz[dic_composites[5]], axis=0)

    composites_matrix = np.zeros((5, day_0.shape[1], day_0.shape[2]))
    for i, matriz_i in enumerate([day_12, day_8, day_4, day_0, day_5_post]):
        composites_matrix[i,:,:] = matriz_i
    
    return composites_matrix, composites_matrix_complete



def calculate_composites_event_3days(pos_HW, matriz):
    import numpy as np
    dic_composites = {}

    time_lags = np.arange(-20, 21, 1)
    for num in time_lags:
        dic_composites[num] = []  
        dic_composites[num].append(pos_HW+num)  

    composites_matrix_complete =  np.zeros((len(time_lags), matriz.shape[1], matriz.shape[2]))
    for ii, lag in enumerate(np.sort(list(dic_composites))):
        composites_matrix_complete[ii] = np.mean(matriz[dic_composites[lag]], axis = 0)

    day_0 = np.mean(matriz[dic_composites[0]], axis=0)
    day_4 = np.mean(matriz[dic_composites[-4]], axis=0)
    day_4_post = np.mean(matriz[dic_composites[5]], axis=0)

    composites_matrix = np.zeros((3, day_0.shape[1], day_0.shape[2]))
    for i, matriz_i in enumerate([day_4, day_0, day_4_post]):
        composites_matrix[i,:,:] = matriz_i
    
    return composites_matrix


=======
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
def calculate_composites2(pos_HW, matriz):
    import numpy as np
    dic_composites = {}

    time_lags = np.arange(-20, 21, 1)
    for pos in pos_HW.index:
        if pos == 0: 
            dic_composites[0] = []
            dic_composites[0].append(int(pos_HW.iloc[0][0]))
            continue
        #elif pos_HW.iloc[pos][0] >= 27270: break  # len(other variables)
        elif pos_HW.iloc[pos][0] == matriz.shape[0]-20: break
        elif pos == pos_HW.index[-1]:
            break
        elif pos_HW.iloc[pos][0]-1 != pos_HW.iloc[pos-1][0]:
            for num in time_lags:  
                if num in dic_composites.keys(): dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))  
                else: 
                    dic_composites[num] = []
                    dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))

    composites_matrix_complete =  np.zeros((len(time_lags), matriz.shape[1], matriz.shape[2]))
    for ii, lag in enumerate(np.sort(list(dic_composites))):
        composites_matrix_complete[ii] = np.mean(matriz[dic_composites[lag]], axis = 0)

    day_0 = np.mean(matriz[dic_composites[0]], axis=0)
    day_5 = np.mean(matriz[dic_composites[-5]], axis=0)
    day_10 = np.mean(matriz[dic_composites[-10]], axis=0)
    day_15 = np.mean(matriz[dic_composites[-15]], axis=0)
    day_20 = np.mean(matriz[dic_composites[-20]], axis=0)
    day_5_post = np.mean(matriz[dic_composites[5]], axis=0)

    composites_matrix = np.zeros((6, day_0.shape[0], day_0.shape[1]))
    for i, matriz_i in enumerate([day_20, day_15, day_10, day_5, day_0, day_5_post]):
        composites_matrix[i,:,:] = matriz_i
    
    return composites_matrix, composites_matrix_complete


# ================================= PHASE SPEED ===============================================
def zonal_mean_1D(temp_mean_2D, Lats, pos_max, path):
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=[7, 5.5])
    plt.plot(np.nanmean(temp_mean_2D, axis=1), color='k')
    plt.ylabel('Mean zonal wind [m/s]', fontsize=13)
    plt.xlabel('Latitude [°]', fontsize=13)
    plt.xticks(np.floor(np.linspace(0, len(Lats)-1, 9)).astype(int), Lats[np.floor(np.linspace(0, len(Lats)-1, 9)).astype(int)], fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(np.nanmean(temp_mean_2D, axis=1)[pos_max], 0, pos_max, color='teal', linestyle='--')
    plt.axvline(pos_max, 0, pos_max, color='teal', linestyle='--')
<<<<<<< HEAD
    plt.savefig(path, dpi=500)
    plt.close()
    
# def zonal_mean_1D_all(array_series_u, Lats, pos_max, path):
#     import matplotlib.colors
#     import matplotlib.pyplot as plt
#     import numpy as np
#     fig = plt.figure(figsize=[7, 5.5])
#     plt.plot(, color='k')
#     plt.ylabel('Mean zonal wind [m/s]', fontsize=13)
#     plt.xlabel('Latitude [°]', fontsize=13)
#     plt.xticks(np.floor(np.linspace(0, len(Lats)-1, 9)).astype(int), Lats[np.floor(np.linspace(0, len(Lats)-1, 9)).astype(int)], fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.axhline(np.nanmean(temp_mean_2D, axis=1)[pos_max], 0, pos_max, color='teal', linestyle='--')
#     plt.axvline(pos_max, 0, pos_max, color='teal', linestyle='--')
#     plt.savefig(path, dpi=200)
#     plt.close()

# ================================= EOF ===============================================
def maps_EOF(Matriz, lats, lons, min, max, cmap, pcvars, path, var = ''):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[15, 21])
    for i, tit in enumerate(['EOF1', 'EOF2', 'EOF3', 'EOF4', 'EOF5']):
        ax = fig.add_subplot(5, 1, i + 1, projection=crs.PlateCarree(central_longitude=180))
        #ax.outline_patch.set_edgecolor('None')

        #ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

        im = ax.contourf(lons, lats, Matriz[i, :, :], cmap=cmap, extend='both',
                         levels=np.arange(min, max, 0.005), transform=crs.PlateCarree())

        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                          linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

        gl.top_labels = False
        gl.right_labels = False
        #plt.yticks(lats[np.where([ii in [30, 60, 90] for ii in lats])[0]], ['30', '60', '90'], fontsize=13, color='dimgray')
        plt.yticks(fontsize=13, color='dimgray')
        gl.xlabel_style = {'size': 14, 'color': 'dimgrey'}
        gl.ylabel_style = {'size': 14, 'color': 'dimgrey'}
        ax.set_title(f'{tit} - {round(pcvars[i],2)}%', fontsize=15, color='k')
    cbaxes = fig.add_axes([0.3, 0.06, 0.4, 0.015])
    cb = plt.colorbar(im, orientation="horizontal", pad=0.2, cax=cbaxes, format='%.3f')
    #cb.set_label(r'PSI300 anomalies [$10^8 m^2 s^{-1}$]', fontsize=15, color='dimgrey')
    cb.set_label(var, fontsize=15, color='dimgrey')
    cb.outline.set_edgecolor(None)
    cb.ax.tick_params(labelcolor='dimgrey', color='dimgrey', labelsize=14)
    plt.axis('tight')
    #plt.show()
    plt.savefig(path, dpi=200)
    plt.close()
=======
    plt.savefig(path, dpi=200)
    plt.close()
    
>>>>>>> ebc5917829b26410676fca5c94adc1a9cd81ec6e
