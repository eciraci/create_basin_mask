#!/bin/sh
python convert_shp_to_raster.py './data/input/Indus.dir/Indus.shp'  --res=0.5
python convert_raster_to_shp.py './data/output/shapefile_to_raster/Indus/Indus.tiff'
