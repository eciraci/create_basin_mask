# - Python Dependencies
from __future__ import print_function
import os
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import rasterio
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from utility_functions import create_dir
# - Change Default Matplotlib Settings
plt.rc('font', family='monospace')
plt.rc('font', weight='bold')
plt.style.use('seaborn-deep')


def convert_shp_to_raster():
    pass


def main():
    parser = argparse.ArgumentParser(
        description="""- Compute Input Binary Mask at the selected 
        resolution."""
    )
    parser.add_argument('input_data_path', nargs=1, default=os.getcwd(),
                        help='Absolute Path to input data.')

    # - Output directory
    parser.add_argument('--outdir', '-O',
                        type=str,
                        default=os.path.join(os.getcwd(), 'data', 'output'),
                        help='Output directory.')

    # - Default Coordinate Reference System - EPSG code
    parser.add_argument('--crs', '-T',
                        type=int, default=4326,
                        help='Coordinate Reference System - def. EPSG:4326')

    # - Output Raster Resolution
    parser.add_argument('--res', '-R', type=float,
                        default=0.5,
                        help='Output Raster Resolution.')

    # - Output Mask Domain Grid
    parser.add_argument('--boundaries', '-B', type=str,
                        default='-90,90,-180,180',
                        help='Domain BBOX (WGS84) - Y-Min, Y-Max,'
                             'X-Min, X-Max')

    args = parser.parse_args()

    # -
    print('# - Input data: {}'.format(args.input_data_path[0]))

    # - Read Input Shapefile with GeoPandas.
    gdz_df = gpd.read_file(args.input_data_path[0]).to_crs(epsg=args.crs)
    # - extract dataset name
    basin_name \
        = args.input_data_path[0].split(os.path.sep)[-1].replace('.shp', '')

    # - Define Output Resolution Grid.
    domain_bbox = args.boundaries.split(',')
    y_min = float(domain_bbox[0])
    y_max = float(domain_bbox[1])
    x_min = float(domain_bbox[2])
    x_max = float(domain_bbox[3])

    print('# - Selected Coordinate Reference System - EPSG:{}'.format(args.crs))
    print('# - Output Domain Limits: ')
    print('# - Y-Min: {}'.format(y_min))
    print('# - Y-Max: {}'.format(y_max))
    print('# - X-Min: {}'.format(x_min))
    print('# - X-Max: {}'.format(x_max))

    x_vect = np.arange(x_min, x_max+args.res, args.res)
    y_vect = np.arange(y_min, y_max+args.res, args.res)
    xx, yy = np.meshgrid(x_vect, y_vect)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    # - Find intersection between Domain Grid Point and
    # - Input basin boundaries using GeoPandas Spatial Join.
    xy_point = gpd.GeoSeries([Point(x, y)
                              for x, y in list(zip(xx_flat, yy_flat))])
    df_pt = gpd.GeoDataFrame({'geometry': xy_point,
                              'df_pt': np.arange(len(xx_flat)),
                              'x': xx_flat, 'y': yy_flat})\
        .set_crs(epsg=args.crs, inplace=True)
    gdz_df_inter = gpd.sjoin(df_pt, gdz_df, predicate='within')
    ind_int = list(gdz_df_inter['df_pt'])
    # - Save the obtained mask
    out_xx = np.array(xx_flat)[ind_int]
    out_yy = np.array(yy_flat)[ind_int]

    # - Initialize Output Binary Maks
    out_bin_mask = np.zeros(xx.shape)
    for k in range(0, len(out_yy)):
        out_bin_mask[np.where(y_vect == out_yy[k]),
                     np.where(x_vect == out_xx[k])] = 1.

    # - create a new xarray containing the  cropped velocity map
    if args.crs == 4326:
        dset_mask = xr.Dataset(data_vars=dict(
            mask=(["y", "x"], out_bin_mask)),
            coords=dict(lat=(["y", "x"], yy),
                        lon=(["y", "x"], xx))
        )

        dset_mask['lon'].attrs['units'] = 'degree east'
        dset_mask['lon'].attrs['long_name'] = 'Longitude'
        dset_mask['lon'].attrs['actual_range'] = [np.min(xx), np.max(xx)]
        dset_mask['lon'].attrs['standard_name'] = 'longitude'
        dset_mask['lon'].attrs['coordinate_defines'] = 'point'

        dset_mask['lat'].attrs['units'] = 'degree north'
        dset_mask['lat'].attrs['long_name'] = 'Latitude'
        dset_mask['lat'].attrs['actual_range'] = [np.min(yy), np.max(yy)]
        dset_mask['lat'].attrs['standard_name'] = 'latitude'
        dset_mask['lat'].attrs['coordinate_defines'] = 'point'

    else:
        dset_mask = xr.Dataset(data_vars=dict(
            mask=(["y", "x"], out_bin_mask)),
            coords=dict(north=(["y", "x"], yy),
                        east=(["y", "x"], xx))
        )

    dset_mask.attrs['EPSG'] = args.crs
    dset_mask.attrs['resolution'] = str(args.res)
    dset_mask.attrs['Y_min'] = y_min
    dset_mask.attrs['Y_max'] = y_max
    dset_mask.attrs['X_max'] = x_max
    dset_mask.attrs['X_min'] = x_max

    # - create output directory
    out_dir = create_dir(args.outdir, basin_name)
    # - save the cropped velocity field
    dset_mask.to_netcdf(os.path.join(out_dir, basin_name+'.nc'),
                        format="NETCDF4")
    # -
    print('# - Process Completed.')
    print('# - Output Mask is available at this path:')
    print('# -> ' + os.path.join(out_dir, basin_name+'.nc'))


# -- run main program
if __name__ == '__main__':
    main()
