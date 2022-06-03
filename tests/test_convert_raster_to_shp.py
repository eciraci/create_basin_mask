#!/usr/bin/env python
u"""
test_convert_raster_to_shp.py
Written by Enrico Ciraci' (06/2022)
Preliminary testo of test_convert_raster_to_shp.py
"""
import os
import pytest
from pydantic import ValidationError
from convert_raster_to_shp import convert_raster_to_shapefile


data_path = './data/output/shapefile_to_raster/Indus/Indus.tiff'
out_dir = os.path.join(os.getcwd(), 'data', 'output', 'raster_to_shapefile')


def test_conversion_shp_to_raster() -> None:
    mask = convert_raster_to_shapefile(data_path, out_dir)
    assert isinstance(mask, str)


def test_valid_path() -> None:
    with pytest.raises(FileNotFoundError):
        convert_raster_to_shapefile(data_path, 1)


def test_validation_error() -> None:
    with pytest.raises(ValidationError):
        convert_raster_to_shapefile(data_path, out_dir, res=0.5)
