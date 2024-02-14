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
plot the region that we are working with
"""

## plot the nlfd
canada_layer = gpd.read_file('Z:/_CanadaLayers/Vectors/canada.shp').to_crs(nlfd_crs.crs)
can = canada_layer.plot(color = 'green')
nlfd_crs.plot(ax = can, color = 'blue')

## add the extent of the disturbance raster to the plot
dist_extent = dist.rio.bounds()

from shapely.geometry import box
bxxed =  box(*dist_extent)
gdf = gpd.GeoDataFrame(index=[0], crs = dist.rio.crs, geometry=[bxxed])
gdf.plot(ax=can, color='grey')

plt.show()
"""
Mask the disturbance raster to the season of interest
"""
spring_gyc = mask_season(nlfd_crs, dist, [4,5,6,7])
fall_gyc = mask_season(nlfd_crs, dist, [8,9,10,11])

"""
For every year in the disturbance raster, count the number of pixels that are disturbed by season
"""
years_range = range(1986, 2023)
def mask_season(nlfd_in, dist, years_range):
    counts_years = []
    for year in years_range:
        #filter shapefile to the year of interest and month of interest 
        nlfd_year = nlfd_in[nlfd_in['YEAR'].isin([year])]
        early_fire = nlfd_year[nlfd_year['MONTH'].isin([5,6])]
        late_fire = nlfd_year[nlfd_year['MONTH'].isin([8,9])]
        # turn this into a raster with values from YEAR in the shapefile
        # USing geocube turn the nlfd_season into a raster
        nlfd_early = make_geocube(vector_data=early_fire, 
                                   measurements=["YEAR"], like=dist)
        nlfd_late = make_geocube(vector_data=late_fire, 
                                   measurements=["YEAR"], like=dist)
        # convert to a xarray dataarray and round values to the nearest year
        nlfd_early = nlfd_early.YEAR.round()
        # convert to a xarray dataarray and round values to the nearest year
        nlfd_late = nlfd_late.YEAR.round()
        #Mask out values in dist that are not equal to the year in nlfd_year
        dist_early_year = dist.where(nlfd_early.notnull(), other = np.nan)
        dist_late_year = dist.where(nlfd_late.notnull(), other = np.nan)
        ## ccount the total number of pixels in the raster that are equal to the year
        count = np.count_nonzero(~np.isnan(dist_early_year.values))
        count2 = np.count_nonzero(~np.isnan(dist_late_year.values))
        counts_years.append([year, count, count2])
        # dist_early_year.rio.to_raster(out + '/GYC_masks/UTM'+ utm + year+'early.tif')
        # dist_late_year.rio.to_raster(out + '/GYC_masks/Years/UTM'+ utm + year + 'late.tif')
    return counts_years
test_ = mask_season(nlfd_crs, dist, years_range)

##get the area of early and late fire years by yrea 

def area_season(nlfd_in, years_range):
    counts_years = []
    for year in years_range:
        #filter shapefile to the year of interest and month of interest 
        nlfd_year = nlfd_in[nlfd_in['YEAR'].isin([year])]
        early_fire = nlfd_year[nlfd_year['MONTH'].isin([5,6])]
        ## change to also include fires that started after July 15th
        late_fire = nlfd_year[nlfd_year['MONTH'].isin([7, 8, 9])]
        ## subset to those that started after July 15th
        late_fire = late_fire[(late_fire['MONTH'] == 7) & (late_fire['DAY'] > 5) | (late_fire['MONTH'].isin([8, 9]))]
        # calculate the area in early and late fire years
        total_area = nlfd_year.area.sum()
        area_early = early_fire.area.sum()
        area_late = late_fire.area.sum()
        counts_years.append([year, total_area, area_early, area_late])
    return counts_years
test_ = pd.DataFrame(area_season(nlfd_crs, years_range))
test_.iloc[:, 1:3] = test_.iloc[:, 1:3].rolling(window=3).mean()

## plot the total area of early and late fire years by year
## plot as a bar chart with a separate bar for each variable 
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
test_.plot(x=0, y=[2, 3], kind='line', ax=ax)
## add labels that line 2 is EARLY and line 3 is LATE
plt.legend(['Early Summer', 'Late Summer'])
plt.ylabel('Area Burned (ha)')
plt.xlim(2000, 2020)
plt.xlabel('Year')
plt.show()

# get the anems of nlfd
nlfd.columns
##get the first few values of the column 'OUT_DATE'
nlfd['OUT_DATE'].head()

# load raster that describes the rate of regrowth
regrowth = rioxarray.open_rasterio(source + '/UTM_' + utm
                                    +'/Results/Change_metrics/SRef_10S_PostChange_evolution_rate.dat')

#calculate average regrowth for each season
spring_regrowth = regrowth.where(dist_mask.notnull(), other = np.nan)
#spring_regrowth.plot()
fall_regrowth = regrowth.where(fall_gyc.notnull(), other = np.nan)
#write the rasters to a file 
spring_regrowth.rio.to_raster(out + '/GYC_masks/UTM'+ utm + 'spring_regrowth.tif')
fall_regrowth.rio.to_raster(out + '/GYC_masks/UTM'+ utm + 'fall_regrowth.tif')


## read in our rasters 
spring_regrowth = rioxarray.open_rasterio(out + '/GYC_masks/UTM'+ utm + 'spring_regrowth.tif')
fall_regrowth = rioxarray.open_rasterio(out + '/GYC_masks/UTM'+ utm + 'fall_regrowth.tif')

#turn the rasters into numpy arrays
spring_regrowth_v = spring_regrowth.values    
fall_regrowth_v = fall_regrowth.values
#remove the nan values
spring_regrowth_v = spring_regrowth_v[~np.isnan(spring_regrowth_v)]
fall_regrowth_v = fall_regrowth_v[~np.isnan(fall_regrowth_v)]
#print the dimensions of the arrays
spring_regrowth.shape
fall_regrowth.shape

#compare the values of the two seasons  
spring_regrowth.mean()
spring_regrowth.std()
fall_regrowth.mean()
fall_regrowth.std()

#plot the histograms of VALUES for the two seasons on the same plot
import matplotlib.pyplot as plt
#create subplots
fig, ax = plt.subplots()
#plot the histograms
ax.hist(spring_regrowth_v, bins = 100, alpha = 0.5, label = 'Spring')
ax.hist(fall_regrowth_v, bins = 100, alpha = 0.5, label = 'Fall')
#add a legend
ax.legend()
#add a title
ax.set_title('Regrowth Rate by Season')
#add labels
ax.set_xlabel('Regrowth Rate')
ax.set_ylabel('Frequency')
#show the plot
plt.show()

#two unique colors in a vector as RGBA
cols = ['seagreen', 'orange']
fig, ax = plt.subplots()
#create boxplot and clip the outliers
bp = ax.boxplot([spring_regrowth_v, fall_regrowth_v], 
           patch_artist = True,
           showfliers = False)
## color by the different seasons
for i in bp['boxes']:
    for patch, color in zip(bp['boxes'], cols):
        patch.set_facecolor(color)
ax.set_xticklabels(['Spring', 'Fall'])
ax.set_ylabel('Regrowth Rate')
ax.set_title('Regrowth Rate by Season')
plt.show()

"""
testing areas

"""
import random
import matplotlib.pyplot as plt 
rast = fall_gyc
# Get the bounds of the raster
bounds = rast.rio.bounds()
bounds = nlfd.total_bounds
# Generate a random bounding box within the raster bounds
# Assuming the raster is large enough, adjust the size of the box as needed
box_size = 10000  # Size of the box in degrees, adjust as needed

min_x = random.uniform(bounds[0], bounds[2] - box_size)
min_y = random.uniform(bounds[1], bounds[3] - box_size)
max_x = min_x + box_size
max_y = min_y + box_size


# Crop the raster to the random bounding box
random_area = rast.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
random_area = nlfd.cx[min_x:max_x, min_y:max_y]
# Plot the cropped area
random_area.plot()
plt.show()