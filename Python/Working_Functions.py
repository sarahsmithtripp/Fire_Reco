"""
This script is the main working functions for the code workflow.
all other scripts source from here.
Running function list is here: 
1. mask_season - this will mask the disturbance raster to the season of interest based on data fraom the nlfd database
2. UPDATE THIS SCRIPT PLEASE GOD 
"""
"""
Load in the libraries
"""
import geopandas as gpd
import rioxarray
import numpy as np
from geocube.api.core import make_geocube

"""
First important function: 
1. nlfd_in - the input shapefile of the national large fire database
2. dist - the disturbance raster
3. season - the season of interest (e.g. [4,5,6,7] for spring)
"""
def mask_season(nlfd_in, dist, season):
    #filter shapefile to a subset of months defined in the function
    nlfd_season = nlfd_in[nlfd_in['MONTH'].isin(season)]
    # turn this into a raster with values from YEAR in the shapefile
    # USing geocube turn the nlfd_season into a raster
    nlfd_season = make_geocube(vector_data=nlfd_season, 
                               measurements=["YEAR"], like=dist)
    # convert to a xarray dataarray and round values to the nearest year
    nlfd_year = nlfd_season.YEAR.round()
    #Mask out values in dist that are not equal to the year in nlfd_year
    dist_mask = dist.where(dist == nlfd_year)
    return dist_mask


"""
*** MASKING MONTHS ***
Function Parameters:

file_list: A list of filenames representing raster data.
months: A list of integers representing months for which metrics are to be calculated.
Processing Steps:

The function iterates over each file in the file_list.
For each file:
It constructs a path to open the raster file.
Extracts the name from the file (excluding extensions and prefixes).
Initializes an empty DataFrame to store the calculated metrics.
Iterates over each month in the list of months:
Opens a mask file for disturbance for the specific month.
Masks the metric raster data with the month's mask.
Converts the raster data to a NumPy array and removes NaN values.
Calculates the mean, standard deviation, and count of the non-NaN values.
Appends these metrics along with the month and name to the DataFrame.
Appends the DataFrame to a list of metrics for all files.
Concatenates all metric DataFrames into a single DataFrame.
Returns the concatenated DataFrame containing all calculated metrics.
Dependencies:

The code utilizes external libraries such as rioxarray for handling raster data and pandas for DataFrame operations.
Output:

The function returns a DataFrame containing metrics for each file and month provided in the input parameters.
"""

def month_metrics(file_list, months):
    metrics_out = []
    for file in file_list:
        print(source + '/UTM_' + utm + '/Results/Change_metrics/' + file)
        metric = rioxarray.open_rasterio(source + '/UTM_' + utm + '/Results/Change_metrics/' + file)
        #get the name of file (without the .dat and SRef_10S_ prefix)
        name = file.split('.')[0].split('_')[2]
        # create an empty dataframe
        metric_df = pd.DataFrame(columns=['Month', 'Mean', 'Std', 'Count', 'Name'])
        # for each month in the list of months mask the regrowth raster to the month
        # and calculate the mean, standard deviation and count of the values
        for month in months:
            #read in the mask for disturbance 
            mask = rioxarray.open_rasterio(out + '/GYC_masks/months/'+ utm + 'month_' + str(month) + '.tif')
            #mask metric raster to the month mask 
            metric_month = metric.where(mask.notnull(), other = np.nan)
            #turn the raster into a numpy array
            metric_month_v = metric_month.values
            #remove the nan values
            metric_month_v = metric_month_v[~np.isnan(metric_month_v)]
            #calculate the mean, standard deviation and count of the values
            mean = metric_month_v.mean()
            std = metric_month_v.std()
            count = metric_month_v.size
            #add these values to the dataframe
            metrics_array = np.array([month, mean, std, count, name])
            metric_df.loc[len(metric_df)] = metrics_array
        #add the dataframe to a list for all metrics
        metrics_out.append(metric_df)
    # concatenate all metric dataframes into a single dataframe
    metrics_out = pd.concat(metrics_out, ignore_index=True)
    return metrics_out

"""
***MASKING SEASONS***
"""
def season_metrics(file_list, seasons):
    metrics_out = []
    for file in file_list:
        print(source + '/UTM_' + utm + '/Results/Change_metrics/' + file)
        metric = rioxarray.open_rasterio(source + '/UTM_' + utm + '/Results/Change_metrics/' + file)
        #get the name of file (without the .dat and SRef_10S_ prefix)
        name = file.split('.')[0].split('_')[2]
        # create an empty dataframe
        metric_df = pd.DataFrame(columns=['Season', 'Mean', 'Std', 'Count', 'Name'])
        # for each month in the list of months mask the regrowth raster to the month
        # and calculate the mean, standard deviation and count of the values
        for season in seasons:
            #read in the mask for disturbance 
            mask = rioxarray.open_rasterio(out + '/GYC_masks/months/'+ utm + 'month_' + str(month) + '.tif')
            #mask metric raster to the month mask 
            metric_month = metric.where(mask.notnull(), other = np.nan)
            #turn the raster into a numpy array
            metric_month_v = metric_month.values
            #remove the nan values
            metric_month_v = metric_month_v[~np.isnan(metric_month_v)]
            #calculate the mean, standard deviation and count of the values
            mean = metric_month_v.mean()
            std = metric_month_v.std()
            count = metric_month_v.size
            #add these values to the dataframe
            metrics_array = np.array([month, mean, std, count, name])
            metric_df.loc[len(metric_df)] = metrics_array
        #add the dataframe to a list for all metrics
        metrics_out.append(metric_df)
    # concatenate all metric dataframes into a single dataframe
    metrics_out = pd.concat(metrics_out, ignore_index=True)
    return metrics_out