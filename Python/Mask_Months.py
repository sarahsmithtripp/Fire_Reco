import rioxarray
import geopandas as gpd
import numpy as np
from geocube.api.core import make_geocube
import pandas as pd
"""
source the working functions script 
"""
exec(open('F:/Fire_Recovery/Code_Workflow/Python/Working_Functions.py').read())
#check that maks_season is in the namespace return error if not
'mask_season' in locals()
"""
Read in large fire database and the disturbance raster
"""
#save location 
out = 'F:/Fire_Recovery'
#load national large fire database (fires > 200 ha)
nlfd = gpd.read_file('F:/Fire_Recovery/NLFD/NFDB_poly_20210707_large_fires.shp')
# read in dist raster (NTEMS Greatest Change Year 1986-2022)
utm = '10S'
source = '//Frst-frm-2232b/l/C2C_1984_2022'
dist = rioxarray.open_rasterio(source + '/UTM_' + utm +
                               '/Results/Change_metrics/SRef_10S_Greatest_Change_Year.dat')
years = np.unique(dist.values)
#change the coordinate system of the shapefile to match the raster
nlfd_crs = nlfd.to_crs(dist.rio.crs)
# check the crs of the shapefile and the raster are the same 
nlfd_crs.crs == dist.rio.crs

"""
Mask to the months of the raster
"""
# create a list of months from 1 to 12
months = list(range(1,13))
# #for each month in this list create a mask using the mask_season function
# #and save the mask to a file
# for month in months:
#     mask = mask_season(nlfd_crs, dist, [month])
#     mask.rio.to_raster(out + '/GYC_masks/months/'+ utm + 'month_' + str(month) + '.tif')

"""
REGROWTH 
Create Monthly Averages, standard deviations and counts in a tidy dataframe
"""
 #create an empty dataframe
recovery_by_month = pd.DataFrame()

# for each month in the list of months mask the regrowth raster to the month 
# and calculate the mean, standard deviation and count of the values
for month in months:
    #read in the mask for disturbance 
    mask = rioxarray.open_rasterio(out + '/GYC_masks/months/'+ utm + 'month_' + str(month) + '.tif')
    #mask the regrowth raster to the month
    regrowth = rioxarray.open_rasterio(source + '/UTM_' + utm
                                    +'/Results/Change_metrics/SRef_10S_PostChange_evolution_rate.dat')
    regrowth_month = regrowth.where(mask.notnull(), other = np.nan)
    #turn the raster into a numpy array
    regrowth_month_v = regrowth_month.values
    #remove the nan values
    regrowth_month_v = regrowth_month_v[~np.isnan(regrowth_month_v)]
    #calculate the mean, standard deviation and count of the values
    mean = regrowth_month_v.mean()
    std = regrowth_month_v.std()
    count = regrowth_month_v.size
    #add these values to the dataframe
    recovery_by_month = recovery_by_month.concat({'Month': month, 'Mean': mean,
                                                   'Std': std, 'Count': count}, 
                                                   ignore_index=True)


recovery_by_month["SEM"] = recovery_by_month["Std"] / np.sqrt(recovery_by_month["Count"])

# create a grouped bar plot of the mean with  error bars of standard deviation
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
recovery_by_month.plot(x='Month', y='Mean', kind='bar', yerr='SEM', ax=ax)
## label the y axis
plt.ylabel('Mean Regrowth Rate')
plt.show()

"""
R
Create Monthly Averages, standard deviations and counts in a tidy dataframe
"""
# create a list of file types that end in .dat from this directory
import os
file_list = os.listdir(source + '/UTM_' + utm + '/Results/Change_metrics/')
file_list = [f for f in file_list if f.endswith('.dat')]

#run the function
metrics = month_metrics(file_list, months)

## get the distinct names of the metrics
names = metrics['Name' ].unique()
# remote 'LastChange' from the list
names = names[names != 'First|Greatest|Last| ']
# remove empty from list
names

# for the file list rep each element 12 times and create a new list 
# of file names
file_name = [file for file in file_list for i in range(12)]
file_name = list(map(lambda x: x.split('.')[0].split('10S_')[1], file_name))
metrics['file_name'] = file_name
## write metrics to a csv
metrics.to_csv('F:/Fire_Recovery/GYC_Masks/months/metrics.csv')

## get the distinct names of the metrics
names = metrics['file_name'].unique()
# remove all names that start with 'First|Greatest|Last| '
names_todrop = ['First', 'Greatest', 'Last', ' ']
names_sub = list(filter(lambda x: not x.startswith(tuple(names_todrop)), names))
# return just the unique elements of the list
names_sub = list(set(names_sub))
len(names_sub)

import matplotlib.pyplot as plt

plots = []  # List to store the plots

for name in names_sub:
    subset = metrics[metrics['file_name'] == name][1:12]
    subset['Month'] = pd.to_numeric(subset['Month'])
    subset['Count'] = pd.to_numeric(subset['Count'])
    subset['Mean'] = pd.to_numeric(subset['Mean'], errors='coerce')
    subset['Std'] = pd.to_numeric(subset['Std'], errors='coerce')
    subset["SEM"] = subset["Std"] / np.sqrt(subset["Count"])
    
    fig, ax = plt.subplots()
    # make the bars light grey with a black edge
    ax.bar(subset['Month'], subset['Mean'], yerr=subset['SEM'], color='lightgrey', edgecolor='black')
    ax.set_ylabel('Mean ' + name)
    ax.set_xlabel('Month')
    
    plots.append(fig)  # Store the plot in the list
    
## make an empty dataframe with two columns, month and metric
metrics = pd.DataFrame(columns = ['Month', 'Metric'])

s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['c', 'd'])
metrics_out = s1
metrics_out = pd.concat([s1, s2])

## 
test = pd.concat([metrics,{12,12}],
                            ignore_index=True)
metric_df = pd.DataFrame(