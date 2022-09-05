#!/usr/bin/env python
u"""
test_convert_raster_to_shp.py
Written by Enrico Ciraci' (06/2022)
Preliminary test of test_convert_raster_to_shp.py
"""
import os
import pytest
from utility_functions import create_dir


# - Path to not existing directory
test_dir = './data/test'
data_path = './data/output'


def test_create_dir():
    with pytest.raises(FileNotFoundError):
        create_dir(test_dir, 'data_dir')


def test_new_dir_path():
    new_dir = create_dir(data_path, 'shapefile_to_raster')
    assert os.path.isdir(new_dir)
