# - Python Dependencies
from __future__ import print_function
import os
import argparse
import numpy as np
from datetime import datetime
import rasterio
from rasterio.transform import Affine
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from utility_functions import create_dir
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

print('# - ALL Dependencies imported.')