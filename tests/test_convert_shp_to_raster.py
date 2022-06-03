#!/usr/bin/env python
u"""
test_convert_shp_to_raster.py
Written by Enrico Ciraci' (06/2022)
Preliminary testo of test_convert_shp_to_raster.py
"""
from convert_shp_to_raster import convert_shp_to_raster
import os
import pytest


data_path = './data/input/Indus.dir/Indus.shp'
out_dir = os.path.join(os.getcwd(), 'data', 'output', 'shapefile_to_raster')


def test_conversion_shp_to_raster():
    mask = convert_shp_to_raster(data_path, out_dir, '-90,90,-180,180')
    assert isinstance(mask, dict)


def test_dtype_conversion_to_raster():
    with pytest.raises(IndexError):
        convert_shp_to_raster(data_path, out_dir, 1)


def test_valid_path():
    with pytest.raises(FileNotFoundError):
        convert_shp_to_raster(data_path, 1, '-90,90,-180,180')
