# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:35:02 2024

@author: green
"""



import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
from shapely.geometry import Point
from geopy.distance import great_circle
from tqdm import tqdm

MIN_INFO = np.load('D:/MBL2024.03.14/data/earthquake/MIN_INFO.npy')
# %%
# Earthquake global minimum distace
stacks_info = np.vstack(MIN_INFO)
GD = pd.DataFrame(columns = ['num','latitude','longitude','min_distance'],index = np.arange(len(stacks_info)))
GD[: ] =stacks_info
GD = GD.astype('float')


# %% 50km 이하,100km 이하 list
GD_50 = GD[GD.min_distance<=50]
GD_100 = GD[GD.min_distance<=100]

# %% Earthquake global map 50
import os
import numpy as np
from netCDF4 import Dataset,MFDataset,date2num,num2date
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import pandas as pd
from cartopy.io import shapereader
from copy import deepcopy
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
# %%

index = ['number','time','magnitude','depth','max','latitude','longitude',]
raw_data = pd.read_excel('D:/MBL2024.03.14/data/earthquake/2021_Earthquake_global_(4.5).xlsx',engine='openpyxl',names = index,header=1,sheet_name='Sheet2')

raw_data.set_index(pd.DatetimeIndex(raw_data.time),inplace=True)

# %%
MIN_INFO = np.load('D:/MBL2024.03.14/data/earthquake/MIN_INFO_v2.npy')

# Earthquake global minimum distace
stacks_info = np.vstack(MIN_INFO)
GD = pd.DataFrame(columns = ['num','latitude','longitude','min_distance','magnitude'],index = np.arange(len(stacks_info)))
GD[: ] =stacks_info
GD = GD.astype('float')


# %% 50km 이하,100km 이하 list
GD_30 = deepcopy(GD[GD.min_distance<=30])
GD_50 = deepcopy(GD[GD.min_distance<=50])
GD_100 = deepcopy(GD[GD.min_distance<=100])
GD_200 = deepcopy(GD[GD.min_distance<=200])
GD_500 = deepcopy(GD[GD.min_distance<=500])

# %%
# plt.rcParams['font.size']=20
# PC = ccrs.PlateCarree()

# fig, axs = plt.subplots(2,2,figsize=(35,16),constrained_layout = True,sharey=True,
#                         sharex=True,dpi=200, gridspec_kw={'height_ratios': [1,1],'width_ratios': [1,1]},
#                         subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180) })

# for i,t in enumerate([GD_30,GD_50,GD_100,raw_data]) : 
   
#     p,q = i//2, i%2
#     print(p,q)
#     target= deepcopy(t)
    
#     axs[p,q].set_extent([-180, 180 ,-80,80],crs=PC)
#     axs[p,q].add_feature(cf.LAND,color=[0.65,0.65,0.65],zorder=50)
#     axs[p,q].add_feature(cf.COASTLINE.with_scale("10m"), lw=1,zorder=110)
#     gl1 = axs[p,q].gridlines(crs=PC, draw_labels=False,y_inline=False,x_inline=False,rotate_labels=False,
#                       linewidth=.6, color='k', alpha=0.45, linestyle='-.',xpadding =10,ypadding =15)
#     gl1.top_labels,gl1.bottom_labels = True,False
#     gl1.left_labels,gl1.right_labels = True,False

#     # gl1.xlocator = mticker.FixedLocator(np.linspace(125.5,127.5,5))
#     # gl1.ylocator = mticker.FixedLocator(np.linspace(33,34,5))
#     # gl1.xformatter = LONGITUDE_FORMATTER
#     # gl1.yformatter = LATITUDE_FORMATTER
    
#     gl1.top_labels,gl1.right_labels = True,False
#     gl1.xlabel_style = gl1.ylabel_style = {"size" : 30, "name" : 'Arial'}

    
#     i=(target.magnitude>=4.5)&(target.magnitude<5.0)
#     M = axs[p,q].scatter(target.longitude[i],target.latitude[i],s=30, edgecolor='k',transform=PC,
#                     color='lightblue',zorder=200,label='4 ≤ M < 5')
#     k=(target.magnitude>=5.0)&(target.magnitude<6.0)
#     M = axs[p,q].scatter(target.longitude[k],target.latitude[k],s=50, edgecolor='k',transform=PC,
#                     color='green',zorder=200,label='5 ≤ M < 6')
#     j=(target.magnitude>=6.0)&(target.magnitude<7.0)
#     M = axs[p,q].scatter(target.longitude[j],target.latitude[j],s=70, edgecolor='k',transform=PC,
#                     color='orange',zorder=200,label='6 ≤ M < 7')
#     h=(target.magnitude>=7.0)&(target.magnitude<8.0)
#     M = axs[p,q].scatter(target.longitude[h],target.latitude[h],s=90, edgecolor='k', transform=PC,
#                     color='red', zorder=200, label='7 ≤ M < 8')
    
    
    
# axs[p,q].legend(loc ='lower center', prop={'family':'Arial', 'size':25}, ncols = 4)
# fig.supylabel('Latitude (\u2070)',position = (-0.05,0.5),fontsize=50, fontname='Arial')
# fig.supxlabel('Longitude (\u2070)',position = (0.5,0.2),fontsize=50,fontname='Arial')


# plt.show(); plt.close()

# bbox_to_anchor=(0, -0.5),

# %%


plt.rcParams['font.size']=20
PC = ccrs.PlateCarree()

fig, axs = plt.subplots(2,2,figsize=(35,16),sharey=True,constrained_layout = True,
                        sharex=True,dpi=200, gridspec_kw={'height_ratios': [1,1],'width_ratios': [1,1]},
                        subplot_kw={'projection':ccrs.PlateCarree(central_longitude=160) })

targets = [GD_30,GD_50,GD_100,raw_data]
   
p,q = 0,0
print(p,q)
target= deepcopy(targets[0])

axs[p,q].set_extent([-180, 180 ,-80,80],crs=PC)
axs[p,q].add_feature(cf.LAND,color=[0.65,0.65,0.65],zorder=50)
axs[p,q].add_feature(cf.COASTLINE.with_scale("10m"), lw=1,zorder=110)
gl1 = axs[p,q].gridlines(crs=PC, draw_labels=False,y_inline=False,x_inline=False,rotate_labels=False,
                  linewidth=.6, color='k', alpha=0.45, linestyle='-.',xpadding =10,ypadding =15)
gl1.top_labels,gl1.bottom_labels = True,False
gl1.left_labels,gl1.right_labels = True,False

# gl1.xlocator = mticker.FixedLocator(np.linspace(125.5,127.5,5))
# gl1.ylocator = mticker.FixedLocator(np.linspace(33,34,5))
# gl1.xformatter = LONGITUDE_FORMATTER
# gl1.yformatter = LATITUDE_FORMATTER


gl1.xlabel_style = gl1.ylabel_style = {"size" : 30, "name" : 'Arial'}


i=(target.magnitude>=4.5)&(target.magnitude<5.0)
M = axs[p,q].scatter(target.longitude[i],target.latitude[i],s=30, edgecolor='k',transform=PC,
                color='peru',zorder=200)
k=(target.magnitude>=5.0)&(target.magnitude<6.0)
M = axs[p,q].scatter(target.longitude[k],target.latitude[k],s=50, edgecolor='k',transform=PC,
                color='green',zorder=200)
j=(target.magnitude>=6.0)&(target.magnitude<7.0)
M = axs[p,q].scatter(target.longitude[j],target.latitude[j],s=70, edgecolor='k',transform=PC,
                color='yellow',zorder=200)
h=(target.magnitude>=7.0)&(target.magnitude<8.0)
M = axs[p,q].scatter(target.longitude[h],target.latitude[h],s=90, edgecolor='k', transform=PC,
                color='red', zorder=200)

p,q = 0,1
print(p,q)
target= deepcopy(targets[1])


axs[p,q].set_extent([-180, 180 ,-80,80],crs=PC)
axs[p,q].add_feature(cf.LAND,color=[0.65,0.65,0.65],zorder=50)
axs[p,q].add_feature(cf.COASTLINE.with_scale("10m"), lw=1,zorder=110)
gl1 = axs[p,q].gridlines(crs=PC, draw_labels=False,y_inline=False,x_inline=False,rotate_labels=False,
                  linewidth=.6, color='k', alpha=0.45, linestyle='-.',xpadding =10,ypadding =15)
gl1.top_labels,gl1.bottom_labels = True,False
gl1.left_labels,gl1.right_labels = True,False

# gl1.xlocator = mticker.FixedLocator(np.linspace(125.5,127.5,5))
# gl1.ylocator = mticker.FixedLocator(np.linspace(33,34,5))
# gl1.xformatter = LONGITUDE_FORMATTER
# gl1.yformatter = LATITUDE_FORMATTER

gl1.xlabel_style = gl1.ylabel_style = {"size" : 30, "name" : 'Arial'}


i=(target.magnitude>=4.5)&(target.magnitude<5.0)
M = axs[p,q].scatter(target.longitude[i],target.latitude[i],s=30, edgecolor='k',transform=PC,
                color='peru',zorder=200)
k=(target.magnitude>=5.0)&(target.magnitude<6.0)
M = axs[p,q].scatter(target.longitude[k],target.latitude[k],s=50, edgecolor='k',transform=PC,
                color='green',zorder=200)
j=(target.magnitude>=6.0)&(target.magnitude<7.0)
M = axs[p,q].scatter(target.longitude[j],target.latitude[j],s=70, edgecolor='k',transform=PC,
                color='yellow',zorder=200)
h=(target.magnitude>=7.0)&(target.magnitude<8.0)
M = axs[p,q].scatter(target.longitude[h],target.latitude[h],s=90, edgecolor='k', transform=PC,
                color='red', zorder=200)

p,q = 1,0
print(p,q)
target= deepcopy(targets[2])

axs[p,q].set_extent([-180, 180 ,-80,80],crs=PC)
axs[p,q].add_feature(cf.LAND,color=[0.65,0.65,0.65],zorder=50)
axs[p,q].add_feature(cf.COASTLINE.with_scale("10m"), lw=1,zorder=110)
gl1 = axs[p,q].gridlines(crs=PC, draw_labels=False,y_inline=False,x_inline=False,rotate_labels=False,
                  linewidth=.6, color='k', alpha=0.45, linestyle='-.',xpadding =10,ypadding =15)
gl1.top_labels,gl1.bottom_labels = False,False
gl1.left_labels,gl1.right_labels = True,True

# gl1.xlocator = mticker.FixedLocator(np.linspace(125.5,127.5,5))
# gl1.ylocator = mticker.FixedLocator(np.linspace(33,34,5))
# gl1.xformatter = LONGITUDE_FORMATTER
# gl1.yformatter = LATITUDE_FORMATTER


gl1.xlabel_style = gl1.ylabel_style = {"size" : 30, "name" : 'Arial'}

i=(target.magnitude>=4.5)&(target.magnitude<5.0)
M = axs[p,q].scatter(target.longitude[i],target.latitude[i],s=30, edgecolor='k',transform=PC,
                color='peru',zorder=200)
k=(target.magnitude>=5.0)&(target.magnitude<6.0)
M = axs[p,q].scatter(target.longitude[k],target.latitude[k],s=50, edgecolor='k',transform=PC,
                color='green',zorder=200)
j=(target.magnitude>=6.0)&(target.magnitude<7.0)
M = axs[p,q].scatter(target.longitude[j],target.latitude[j],s=70, edgecolor='k',transform=PC,
                color='yellow',zorder=200)
h=(target.magnitude>=7.0)&(target.magnitude<8.0)
M = axs[p,q].scatter(target.longitude[h],target.latitude[h],s=90, edgecolor='k', transform=PC,
                color='red', zorder=200)

p,q = 1,1
print(p,q)
target= deepcopy(targets[3])


axs[p,q].set_extent([-180, 180 ,-80,80],crs=PC)
axs[p,q].add_feature(cf.LAND,color=[0.65,0.65,0.65],zorder=50)
axs[p,q].add_feature(cf.COASTLINE.with_scale("10m"), lw=1,zorder=110)
gl1 = axs[p,q].gridlines(crs=PC, draw_labels=False,y_inline=False,x_inline=False,rotate_labels=False,
                  linewidth=.6, color='k', alpha=0.45, linestyle='-.',xpadding =10,ypadding =15)
gl1.top_labels,gl1.bottom_labels = False,False
gl1.left_labels,gl1.right_labels = True,True

# gl1.xlocator = mticker.FixedLocator(np.linspace(125.5,127.5,5))
# gl1.ylocator = mticker.FixedLocator(np.linspace(33,34,5))
# gl1.xformatter = LONGITUDE_FORMATTER
# gl1.yformatter = LATITUDE_FORMATTER


gl1.xlabel_style = gl1.ylabel_style = {"size" : 30, "name" : 'Arial'}


i=(target.magnitude>=4.5)&(target.magnitude<5.0)
M = axs[p,q].scatter(target.longitude[i],target.latitude[i],s=30, edgecolor='k',transform=PC,
                color='peru',zorder=200,label='4 ≤ M < 5')
k=(target.magnitude>=5.0)&(target.magnitude<6.0)
M = axs[p,q].scatter(target.longitude[k],target.latitude[k],s=50, edgecolor='k',transform=PC,
                color='green',zorder=200,label='5 ≤ M < 6')
j=(target.magnitude>=6.0)&(target.magnitude<7.0)
M = axs[p,q].scatter(target.longitude[j],target.latitude[j],s=70, edgecolor='k',transform=PC,
                color='yellow',zorder=200,label='6 ≤ M < 7')
h=(target.magnitude>=7.0)&(target.magnitude<8.0)
M = axs[p,q].scatter(target.longitude[h],target.latitude[h],s=90, edgecolor='k', transform=PC,
                color='red', zorder=200, label='7 ≤ M < 8')
    
    
fig.legend(bbox_to_anchor=(0.9,-0.04), prop={'family':'Arial', 'size':50}, ncols = 4, markerscale = 3)
fig.supylabel('Latitude (\u2070)\n ',position = (-0.05,0.5),fontsize=50, fontname='Arial')
fig.supxlabel('\nLongitude (\u2070)',position = (0.5,-1),fontsize=50,fontname='Arial')


plt.show(); plt.close()

