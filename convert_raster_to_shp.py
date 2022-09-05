#!/usr/bin/env python
u"""
convert_raster_to_shp.py
Written by Enrico Ciraci' (01/2022)

Compute Basin Boundaries from Basin Binary Mask.

NOTE: At least for now, if the input binary mask is provided in NetCDF format,
      EPSG:4326 is assumed as the default coordinate reference system.

COMMAND LINE OPTIONS:
usage: convert_raster_to_shp.py [-h] [--outdir OUTDIR] input_data_path

- Compute Basin Boundaries from Basin Binary Mask.

positional arguments:
  input_data_path       Absolute Path to input data.

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR, -O OUTDIR
                        Output directory.


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
import sys
import argparse
from datetime import datetime
import numpy as np
import rasterio
from rasterio import features
import fiona
import fiona.crs
import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib_scalebar.scalebar import ScaleBar
from pydantic import validate_arguments
from utility_functions import load_tiff, load_netcdf, create_dir

# - Change Default Matplotlib Settings
plt.rc('font', family='monospace')
plt.rc('font', weight='bold')
plt.style.use('seaborn-deep')


@validate_arguments()
def convert_raster_to_shapefile(input_data: str, out_dir: str) -> str:
    """
    Compute Borders of the input binary mask provided as GeoTiff/NetCDF file
    :param input_data: absolute path to binary mask
    :param out_dir: absolute path to output directory
    :return:absolute path to output file
    """
    # - extract input file format
    f_suff = input_data.split('.')[-1]
    # - extract dataset name
    basin_name = input_data.split(os.path.sep)[-1].replace('.'+f_suff, '')
    # - Initialize Output attributes
    crs = None
    m_arr = None
    d_transform = None
    width = None
    height = None

    # - Import Raster Data
    if f_suff.lower() in ['tif', 'tiff', 'geotiff']:
        with rasterio.open(input_data, mode='r+') as src:
            # - read band #1
            m_arr = src.read(1, masked=True)
            crs = src.crs
            d_transform = src.transform
            width = src.width
            height = src.height

    elif f_suff.lower() in ['nc', 'netcdf']:
        src = load_netcdf(input_data, data_name='mask', x='lon', y='lat')
        m_arr = src['data']
        crs = fiona.crs.from_epsg('EPSG:4326')
        d_transform = src['transform']
        width = src['width']
        height = src['height']

    else:
        print('# - Unknown output file format selected.')

    # - create output directory
    out_dir = create_dir(out_dir, basin_name)
    # - absolute path to output file name
    out_f_name = os.path.join(out_dir, basin_name+'.shp')
    # - Define output shapefile schema
    schema = {
        'geometry': 'Polygon',
        'properties': [('Name', 'str'), ('time', 'str'),
                       ('Width', 'int'), ('Height', 'int')]
    }

    with fiona.open(out_f_name, mode='w', driver='ESRI Shapefile',
                    schema=schema, crs=crs.to_string()) as poly_shp:
        # - generate valid data binary mask
        # - valid data - msk = 1
        # - not valid data - msk = 0
        msk = np.full(m_arr.shape, 255).astype('float32')
        msk[m_arr == 0] = 0
        # - Use rasterio.features.shapes to get valid data region
        # - boundaries. For more details:
        # - https://rasterio.readthedocs.io/en/latest/api/
        # -         rasterio.features.html
        b_shapes = list(features.shapes(msk,
                                        transform=d_transform))
        # - In several cases, features.shapes returns multiple
        # - polygons. Use only the polygon with the maximum number
        # - of points to delineate the area covered by valid elevation
        # - data.
        poly_vect_len = []
        for shp_bound_tmp in b_shapes:
            poly_vect_len.append(len(shp_bound_tmp[0]
                                     ['coordinates'][0]))
        max_index = poly_vect_len.index(max(poly_vect_len))
        shp_bound = b_shapes[max_index]
        # -
        row_dict = {
            # - Geometry [Polygon]
            'geometry': {'type': 'Polygon',
                         'coordinates': shp_bound[0]['coordinates']},
            # - Properties [based on the schema defined above]
            'properties': {'Name': basin_name,
                           'time': datetime.now().isoformat(),
                           'Width': width, 'Height': height},
        }
        poly_shp.write(row_dict)

    return out_f_name


def main():
    parser = argparse.ArgumentParser(
        description="""- Compute Basin Boundaries from Basin Binary Mask."""
    )
    parser.add_argument('input_data_path', nargs=1, default=os.getcwd(),
                        help='Absolute Path to input data.')

    # - Output directory
    parser.add_argument('--outdir', '-O',
                        type=str,
                        default=os.path.join(os.getcwd(), 'data', 'output'),
                        help='Output directory.')

    args = parser.parse_args()
    # -
    print(f'# - Input data: {args.input_data_path[0]}')

    # - create raster-to-shp directory
    out_dir = create_dir(args.outdir, 'raster_to_shapefile')

    # - Create selected domain binary mask.
    mask_p = convert_raster_to_shapefile(args.input_data_path[0], out_dir)

    # - compare the obtained shapefile with the input binary
    # - extract input file format
    f_suff = args.input_data_path[0].split('.')[-1]
    if f_suff.lower() in ['tif', 'tiff', 'geotiff']:
        ev_mask = load_tiff(args.input_data_path[0])

    elif f_suff.lower() in ['nc', 'netcdf']:
        ev_mask = load_netcdf(args.input_data_path[0], data_name='mask',
                              x='lon', y='lat')
    else:
        print('# - Unknown output file format selected.')
        sys.exit()

    mask = ev_mask['data']  # - binary mask as numpy array
    x_coords = ev_mask['x_coords']  # - x-axis
    y_coords = ev_mask['y_coords']  # - y-axes
    xx_m, yy_m = np.meshgrid(x_coords, y_coords)  # - mask domain mesh-grid

    # - Read Input Shapefile with GeoPandas.
    gdz_df = gpd.read_file(mask_p)
    shp_bounds = gdz_df.bounds.values[0]

    # - Define output figure extent from input mask bounding box.
    map_extent = [np.floor(shp_bounds[0]) - 10, np.ceil(shp_bounds[2]) + 10,
                  np.floor(shp_bounds[1]) - 10, np.ceil(shp_bounds[3]) + 10]

    # - Compare the obtained binary mask with input shapefile.
    fig = plt.figure(figsize=(7, 5), constrained_layout=True)
    # - initialize legend labels
    leg_label_list = []
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    # - Figure title
    ax.set_title('Raster -> Shapefile', weight='bold', loc='left', size=12)
    ax.coastlines()  # - plot coast lines
    # - Set Map Grid
    grid_ln = ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                           y_inline=False, color='k', linestyle='dotted',
                           alpha=0.3)
    grid_ln.top_labels = False
    grid_ln.bottom_labels = True
    grid_ln.right_labels = False
    grid_ln.xlocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[0]) - 3,
                                         np.floor(map_extent[1]) + 3, 5))
    grid_ln.ylocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[2]) - 5,
                                         np.floor(map_extent[3]) + 5, 5))
    grid_ln.xlabel_style = {'rotation': 0, 'weight': 'bold', 'size': 11}
    grid_ln.ylabel_style = {'rotation': 0, 'weight': 'bold', 'size': 11}
    grid_ln.xformatter = LONGITUDE_FORMATTER
    grid_ln.yformatter = LATITUDE_FORMATTER

    if gdz_df.crs.to_string() == 'EPSG:4326':
        # - Add ScaleBar only if working in geographic coordinates
        ax.add_artist(ScaleBar(1, units='deg', dimension='angle',
                               location='lower right', border_pad=1,
                               pad=0.5, box_color='w',
                               frameon=True))

    # - Plot Input Binary Mask
    ax.pcolormesh(xx_m, yy_m, mask, cmap=plt.get_cmap('viridis'))
    leg_label_list.append('Input Binary Mask')
    leg_1 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                               edgecolor='y', facecolor='y',
                               linestyle='-')

    # - Plot Reference Basin Boundaries
    shape_feature = ShapelyFeature(Reader(mask_p).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor='None',
                   edgecolor='g', linestyle='--',
                   linewidth=2)
    leg_label_list.append('Output Basin Boundaries')
    leg_2 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                               edgecolor='g', facecolor='none',
                               linestyle='--')

    # - Add Legend to Map
    ax.legend([leg_1, leg_2], leg_label_list, loc='upper right',
              fontsize=10, framealpha=1,
              facecolor='w', edgecolor='k')
    # - Add Datetime Annotation
    ax.annotate(f'Last Update: {datetime.now().isoformat()}',
                xy=(0.03, 0.03), xycoords='axes fraction',
                size=7, zorder=100,
                bbox=dict(boxstyle='square', fc='w', alpha=0.8))
    # - save output figure
    fig_format = 'jpeg'
    plt.savefig(mask_p.replace('shp', fig_format),
                dpi=200, format=fig_format)
    plt.close()


# -- run main program
if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f'# - Computation Time: {end_time - start_time}')
