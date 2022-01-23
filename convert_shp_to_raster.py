#!/usr/bin/env python
u"""
convert_shp_to_raster.py
Written by Enrico Ciraci' (01/2022)

Compute binary mask of a selected basin which boundaries are provided in the
form of an ESRI shapefile.

COMMAND LINE OPTIONS:
usage: convert_shp_to_raster.py [-h] [--outdir OUTDIR] [--crs CRS] [--res RES]
                    [--boundaries BOUNDARIES]
                    [--f_type {nc,netcdf,GeoTiff,tif,tiff}] input_data_path

positional arguments:
  input_data_path       Absolute Path to input data.

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR, -O OUTDIR
                        Output directory.
  --crs CRS, -T CRS     Coordinate Reference System - def. EPSG:4326
  --res RES, -R RES     Output Raster Resolution.
  --boundaries BOUNDARIES, -B BOUNDARIES
                        Domain BBOX (WGS84) - Y-Min, Y-Max,X-Min, X-Max
  --f_type {nc,netcdf,GeoTiff,tif,tiff}, -F {nc,netcdf,GeoTiff,tif,tiff}
                        Output File format.



PYTHON DEPENDENCIES:
    numpy: package for scientific computing with Python
           https://numpy.org
    matplotlib: Library for creating static, animated, and interactive
           visualizations in Python.
           https://matplotlib.org
    pandas: Python Data Analysis Library
           https://pandas.pydata.org
    geopandas: Python tools for geographic data
           https://geopandas.org/en/stable/
    rasterio: access to geospatial raster data
           https://rasterio.readthedocs.io
    fiona: Fiona reads and writes geographic data files.
           https://fiona.readthedocs.io
    shapely: Manipulation and analysis of geometric objects in the Cartesian
           plane.
           https://shapely.readthedocs.io/en/stable
    datetime: Basic date and time types
           https://docs.python.org/3/library/datetime.html#module-datetime
    xarray: xarray: N-D labeled arrays and datasets in Python
           https://xarray.pydata.org/en/stable
    cartopy: Python package designed for geospatial data processing in order
           to produce maps and other geospatial data analyses.
           https://scitools.org.uk/cartopy
    matplotlib_scalebar: Provides a new artist for matplotlib to display a
           scale bar.
           https://github.com/ppinard/matplotlib-scalebar


UPDATE HISTORY:

"""
# - Python Dependencies
from __future__ import print_function
import os
import argparse
import numpy as np
from datetime import datetime
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
# -
from utility_functions import save_raster, load_tiff

# - Change Default Matplotlib Settings
plt.rc('font', family='monospace')
plt.rc('font', weight='bold')
plt.style.use('seaborn-deep')


def convert_shp_to_raster(input_data: str, out_dir: str,
                          boundaries: str, res: float = 0.5,
                          ref_crs: int = 4326, o_type='nc') -> dict:
    """
    Create binary mask for a region which borders are provided in the
    form of ESRI shapefile polygon
    :param input_data: path to input shapefile
    :param out_dir: absolute path to output directory
    :param boundaries: output raster grid boundaries
    :param res: output raster grid resolution
    :param ref_crs: reference coordinate reference system
    :param o_type: output file format: NetCDF4 or GeoTiff
    :return: python dictionary containing raster data as numpy array
             input raster + ancillary data.
    """
    # - Read Input Shapefile with GeoPandas.
    gdz_df = gpd.read_file(input_data).to_crs(epsg=ref_crs)

    # - extract dataset name
    basin_name = input_data.split(os.path.sep)[-1].replace('.shp', '')

    # - Define Output Resolution Grid.
    domain_bbox = boundaries.split(',')
    y_min = float(domain_bbox[0])
    y_max = float(domain_bbox[1])
    x_min = float(domain_bbox[2])
    x_max = float(domain_bbox[3])

    print('# - Selected Coordinate Reference System - EPSG:{}'.format(ref_crs))
    print('# - Output Domain Limits: ')
    print('# - Y-Min: {}'.format(y_min))
    print('# - Y-Max: {}'.format(y_max))
    print('# - X-Min: {}'.format(x_min))
    print('# - X-Max: {}'.format(x_max))

    x_vect = np.arange(x_min, x_max + res, res)
    y_vect = np.arange(y_min, y_max + res, res)
    xx, yy = np.meshgrid(x_vect, y_vect)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    # - Find intersection between Domain Grid Point and
    # - Input basin boundaries using GeoPandas Spatial Join.
    xy_point = gpd.GeoSeries([Point(x, y)
                              for x, y in list(zip(xx_flat, yy_flat))])
    df_pt = gpd.GeoDataFrame({'geometry': xy_point,
                              'df_pt': np.arange(len(xx_flat)),
                              'x': xx_flat, 'y': yy_flat}) \
        .set_crs(epsg=ref_crs, inplace=True)

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

    # - create output directory
    out_dir = create_dir(out_dir, basin_name)

    if o_type.lower() in ['nc', 'netcdf']:
        # - create a new xarray containing the  cropped velocity map
        if ref_crs == 4326:
            # - if the standard EPSG:4326 - WGS84 - World Geodetic System 1984
            # - is used, name the x and y axes, longitude and latitude.
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

        dset_mask.attrs['EPSG'] = ref_crs
        dset_mask.attrs['resolution'] = str(res)
        dset_mask.attrs['Y_min'] = y_min
        dset_mask.attrs['Y_max'] = y_max
        dset_mask.attrs['X_max'] = x_max
        dset_mask.attrs['X_min'] = x_max

        # - save the cropped velocity field
        output_file = os.path.join(out_dir, basin_name + '.nc')
        dset_mask.to_netcdf(output_file, format="NETCDF4")
        f_out_type = 'nc'

    elif o_type.lower() in ['tif', 'tiff', 'geotiff']:
        output_file = os.path.join(out_dir, basin_name + '.tiff')
        save_raster(np.flipud(out_bin_mask), res, x_vect, y_vect, output_file,
                    ref_crs)
        f_out_type = 'tiff'
    else:
        print('# - Unknown output file format selected.')
        output_file = ''
        f_out_type = None
    # -
    print('# - Process Completed.')
    print('# - Output Mask is available at this path:')
    print('# -> ' + os.path.join(out_dir, basin_name + '.' + o_type))

    return {'output_file': output_file, 'mask': out_bin_mask,
            'res': res, 'x_coords': x_vect, 'y_coords': y_vect,
            'shp_bounds': gdz_df.bounds, 'gdz_df_inter': gdz_df_inter,
            'crs': ref_crs, 'f_out_type': f_out_type}


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

    # - output file format
    parser.add_argument('--f_type', '-F', type=str,
                        default='nc',
                        choices=['nc', 'netcdf', 'GeoTiff', 'tif', 'tiff'],
                        help='Output File format.')

    args = parser.parse_args()
    # -
    print('# - Input data: {}'.format(args.input_data_path[0]))
    # - create shp-to-raster directory
    out_dir = create_dir(args.outdir, 'shapefile_to_raster')

    # - Create selected domain binary mask.
    mask_p = convert_shp_to_raster(args.input_data_path[0], out_dir,
                                   args.boundaries, res=args.res,
                                   ref_crs=args.crs, o_type=args.f_type)

    # - compare the obtained mask with the input shapefile
    if args.f_type in ['GeoTiff', 'tif', 'tiff']:
        ev_mask = load_tiff(mask_p['output_file'])
        mask = ev_mask['data']              # - binary mask as numpy array
        x_coords = ev_mask['x_coords']      # - x-axis
        y_coords = ev_mask['y_coords']      # - y-axes
    else:
        ev_mask = xr.open_dataset(mask_p['output_file'])
        mask = ev_mask['mask']
        x_coords = mask_p['x_coords']
        y_coords = mask_p['y_coords']

    xx, yy = np.meshgrid(x_coords, y_coords)      # - mask domain mesh-grid
    f_out_type = mask_p['f_out_type']             # - output mask file format
    shp_bounds = mask_p['shp_bounds'].values[0]   # - Reference shapefile bounds
    mask_crs = mask_p['gdz_df_inter'].crs

    # - Define output figure extent from input mask bounding box.
    map_extent = [np.floor(shp_bounds[0])-10, np.ceil(shp_bounds[2])+10,
                  np.floor(shp_bounds[1])-10, np.ceil(shp_bounds[3])+10]

    # - Compare the obtained binary mask with input shapefile.
    fig = plt.figure(figsize=(7, 5), constrained_layout=True)
    # - initialize legend labels
    leg_label_list = list()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    # - Figure title
    ax.set_title('Shapefile -> Raster', weight='bold', loc='left', size=12)
    ax.coastlines()     # - plot coast lines
    # - Set Map Grid
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                      y_inline=False, color='k', linestyle='dotted',
                      alpha=0.3)
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.xlocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[0]) - 3,
                                         np.floor(map_extent[1]) + 3, 5))
    gl.ylocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[2]) - 5,
                                         np.floor(map_extent[3]) + 5, 5))
    gl.xlabel_style = {'rotation': 0, 'weight': 'bold', 'size': 11}
    gl.ylabel_style = {'rotation': 0, 'weight': 'bold', 'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    if args.crs == 4326:
        # - Add ScaleBar only if working in geographic coordinates
        ax.add_artist(ScaleBar(1, units='deg', dimension='angle',
                               location='lower right', border_pad=1,
                               pad=0.5, box_color='w',
                               frameon=True))

    # - Plot Reference Basin Boundaries
    shape_feature = ShapelyFeature(Reader(args.input_data_path[0]).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor='None',
                   edgecolor='r', linestyle='--',
                   linewidth=2)
    leg_label_list.append('Basin Boundaries')
    l1 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                            edgecolor='r', facecolor='none',
                            linestyle='--')

    # - Plot Binary Mask
    im = ax.pcolormesh(xx, yy, mask, cmap=plt.get_cmap('viridis'))
    leg_label_list.append('Binary Mask')
    l2 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                            edgecolor='y', facecolor='y',
                            linestyle='-')

    # - Add Legend to Mao
    ax.legend([l1, l2], leg_label_list, loc='upper right',
              fontsize=10, framealpha=1,
              facecolor='w', edgecolor='k')

    # - Add Datetime Annotation
    ax.annotate('Last Update: {}'.format(datetime.now().isoformat()),
                xy=(0.03, 0.03), xycoords="axes fraction",
                size=7, zorder=100,
                bbox=dict(boxstyle="square", fc="w", alpha=0.8))
    # - save output figure
    fig_format = 'jpeg'
    plt.savefig(mask_p['output_file'].replace(f_out_type, fig_format),
                dpi=200, format=fig_format)
    plt.close()


# -- run main program
if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print("# - Computation Time: {}".format(end_time - start_time))
