import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import warnings
from src.aafitrans.aafitrans import find_transform, MaxIterError

test_coo1 = np.array(
    [
        [894.99090576, 353.99679565],
        [781.95367432, 362.0241394],
        [1629.98413086, 175.95497131],
        [1118.9901123, 123.98012543],
        [181.03701782, 1054.97290039],
        [310.97891235, 39.92118073],
        [667.04632568, 138.02412415],
        [1434.02111816, 735.92120361],
        [1277.95361328, 238.97573853],
        [1590.99182129, 1225.98352051],
        [1590.99182129, 1225.98352051],
        [894.99090576, 353.99679565],
        [829.95367432, 462.97573853],
        [1102.97888184, 39.92118073],
        [1531.04638672, 778.02410889],
        [340.97573853, 969.09082031],
        [643.01586914, 639.95495605],
        [813.0369873, 586.02490234],
    ]
)

test_coo2 = np.array(
    [
        [847.27522227, 886.69418715],
        [171.02178651, 733.852688],
        [58.57737079, 733.80692709],
        [718.57211753, 1182.33837963],
        [519.72699635, 1135.29289272],
        [1630.38109026, 1154.74425444],
        [1571.06441488, 84.37033107],
        [1159.69262346, 115.3545713],
        [971.32113532, 357.38851363],
        [1359.00782936, 752.50892002],
        [1181.65727948, 647.95547337],
        [1535.00206298, 973.39172075],
        [985.85774664, 528.25589459],
        [869.42201185, 761.32927133],
        [838.24600721, 641.1771541],
        [319.32331197, 49.65574402],
        [954.79785274, 838.43816374],
        [1152.06161559, 1006.26577518],
    ]
)

expected_result = np.array(
    [
        [-9.44682863e-01, -3.48389869e-01, 1.81624418e03],
        [3.48389869e-01, -9.44682863e-01, 9.09886211e02],
        [0, 0, 1],
    ]
)


def test_MaxIterError():
    with pytest.raises(MaxIterError):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            find_transform(test_coo1[:4], test_coo2[:4])


def test_matching():
    transf, _ = find_transform(test_coo1, test_coo2, min_matches=10, pixel_tolerance=10)
    np.testing.assert_allclose(transf.params, expected_result)


def test_flipped_matching():
    expected_flipped_result = expected_result.copy()
    expected_flipped_result[0, :] = expected_result[0, :] * -1
    test_coo2_flipped = test_coo2.copy()
    test_coo2_flipped[:, 0] = test_coo2_flipped[:, 0] * -1
    transf, _ = find_transform(
        test_coo1, test_coo2_flipped, min_matches=10, pixel_tolerance=10
    )
    np.testing.assert_allclose(transf.params, expected_flipped_result)
