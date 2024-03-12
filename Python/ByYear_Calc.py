import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rioxarray
import dask.array as da
import spectral_recovery as sr 
import xarray as xr
### read in the location of the fires
nlfd = gpd.read_file('F:/Fire_Recovery/NLFD/NFDB_poly_20210707_large_fires.shp')
nlfd_restor = sr.read_restoration_polygons('F:/Fire_Recovery/NLFD/NFDB_poly_20210707_large_fires.shp')
## BAPS are here (list full names)
baps = [os.path.join('F:/NewBap/NBR_new', file) for file in os.listdir('F:/NewBap/NBR_new')]
print(baps)
## check that I can read the first file as a raster

# for every YEAR in  nlfd subset files to that are for the year before and all years after 
for year in nlfd['YEAR'].unique():
    # get the year before 
    year_before = year - 1
    # get the year after 
    year_after = year + 6
    year_seq = list(range(year_before, year_after))
    # get a list of files that include the same years as in years_seq
    baps_year = [file for file in baps if any(str(year) in file for year in year_seq)]
    # check that the files are there 
    print(baps_year)
    # READ AND STACK THESE RASTERS 
    baps_base = rioxarray.open_rasterio(baps_year[0])
    baps_year = [rioxarray.open_rasterio(file) for file in baps_year]
    babs_year = xr.concat(baps_year, dim = 'time')
    #rename band coordinate to NBR
    babs_year = babs_year.assign_coords(band = ['NBR'])
    ## subset to NLFD to the year of interest 
    nlfd_year = nlfd[nlfd['YEAR'] == year]
    # convert the nlfd to the same crs as the baps_year
    nlfd_year = nlfd_year.to_crs(baps_base.rio.crs)
    # check the crs of the shapefile and the raster are the same 
    nlfd_year.crs == baps_base.rio.crs
