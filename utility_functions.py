"""
Enrico Ciraci 01/2022
Utility Functions:
1. create_dir - create directory at the selected absolute path.
2. save_raster - save a numpy raster in GeoTiff format.
3. load_tiff - read GeoTiff raster ad a numpy array.
4. load_netcdf - read raster saved in NetCDF format.

Python Dependencies:
    numpy: package for scientific computing with Python
           https://numpy.org
    rasterio: access to geospatial raster data
       https://rasterio.readthedocs.io
    xarray: xarray: N-D labeled arrays and datasets in Python
           https://xarray.pydata.org/en/stable
"""
# - python dependencies
from __future__ import print_function
import os
import numpy as np
import rasterio
from rasterio.transform import Affine
import xarray as xr


def create_dir(abs_path: str, dir_name: str) -> str:
    """
    Create directory
    :param abs_path: absolute path to the output directory
    :param dir_name: new directory name
    :return: absolute path to the new directory
    """
    dir_to_create = os.path.join(abs_path, dir_name)
    if not os.path.exists(dir_to_create):
        os.mkdir(dir_to_create)
    return dir_to_create


def save_raster(raster: np.ndarray, res: float, x: np.ndarray,
                y: np.ndarray, out_path: str, crs: int,
                nodata: int = -9999) -> None:
    """
    Save the Provided Raster in GeoTiff format
    :param raster: input raster - np.ndarray
    :param res: raster resolution - integer
    :param x: x-axis - np.ndarray
    :param y: y-axis - np.ndarray
    :param crs: - coordinates reference system
    :param out_path: absolute path to output file
    :param nodata: no-data value
    :return: None
    """
    # - Calculate Affine Transformation of the output raster
    y = np.flipud(y)
    shift = res/2
    transform = (Affine.translation(x[0]-shift, y[0]+shift)
                 * Affine.scale(res, -res))
    print(transform)
    with rasterio.open(out_path, 'w', driver='GTiff',
                       height=raster.shape[0],
                       width=raster.shape[1], count=1,
                       dtype=raster.dtype, crs=crs,
                       transform=transform,
                       nodata=nodata) as dst:
        dst.write(raster, 1)


def load_tiff(in_path: str) -> dict:
    """
    Load raster saved in GeoTiff format
    :param in_path: absolute path to input GeoTiff
    :return: python dictionary containing raster data as numpy array
             input raster + ancillary data (X- and Y-axes info + Geo-Transform).
    """
    with rasterio.open(in_path, mode="r+") as src:
        # - read band #1 - DEM elevation in meters
        raster_input = src.read(1).astype(src.dtypes[0])
        # - raster upper-left and lower-right corners
        ul_corner = src.transform * (0, 0)
        lr_corner = src.transform * (src.width, src.height)
        grid_res = src.res
        # -
        shift_x = src.res[0] / 2
        shift_y = src.res[1] / 2
        x_coords = np.arange(ul_corner[0]+shift_x, lr_corner[0]+shift_x,
                             grid_res[0])
        y_coords = np.arange(lr_corner[1]+shift_y, ul_corner[1]+shift_y,
                             grid_res[1])
        if src.transform.e < 0:
            raster_input = np.flipud(raster_input)
        # - Compute New Affine Transform
        transform = (Affine.translation(x_coords[0]+shift_x,
                                        y_coords[0]-shift_y)
                     * Affine.scale(src.res[0], src.res[1]))

        return{'data': raster_input, 'crs': src.crs, 'res': src.res,
               'y_coords': y_coords, 'x_coords': x_coords,
               'transform': transform, 'src_transform': src.transform,
               'width': src.width, 'height': src.height,
               'ul_corner': ul_corner, 'lr_corner': lr_corner,
               'nodata': src.nodata, 'dtype': src.dtypes[0]}


def load_netcdf(in_path: str, data_name='mask', x='x', y='y') -> dict:
    """
    Load raster saved in NetCDF format
    :param in_path: absolute path to input NetCDF archive
    :param data_name:input raster var name
    :param x:x-axis name
    :param y:y-axis name
    :return: python dictionary containing raster data as numpy array
             input raster + ancillary data (X- and Y-axes info + Geo-Transform).
    """
    src = xr.open_dataset(in_path)
    raster_input = src[data_name].values
    x_coords = src[x].values[0, :]
    y_coords = src[y].values[:, 1]
    grid_res_x = np.abs(x_coords[0] - x_coords[1])
    grid_res_y = np.abs(y_coords[0] - y_coords[1])
    width = raster_input.shape[1]
    height = raster_input.shape[0]
    # -
    shift_x = grid_res_x / 2
    shift_y = grid_res_y / 2

    if y_coords[1] < y_coords[0]:
        raster_input = np.flipud(raster_input)
        y_coords = np.flipud(y_coords)

    ul_corner = (y_coords[-1], x_coords[0])
    lr_corner = (y_coords[0], x_coords[-1])

    # - Compute New Affine Transform
    transform = (Affine.translation(x_coords[0]-shift_x, y_coords[0]-shift_y)
                 * Affine.scale(grid_res_x, grid_res_y))

    return {'data': raster_input, 'res': (grid_res_x, grid_res_y),
            'y_coords': y_coords, 'x_coords': x_coords,
            'transform': transform, 'width': width, 'height': height,
            'ul_corner': ul_corner, 'lr_corner': lr_corner,
            'dtype': raster_input.dtype}
