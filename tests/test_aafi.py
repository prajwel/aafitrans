import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import warnings

from src.aafitrans.aafitrans import find_transform, __version__, MaxIterError


test_coo1 = np.array([np.arange(4)] * 2).T
test_coo2 = np.array([np.ones(4)] * 2).T

def test_MaxIterError():
    with pytest.raises(MaxIterError):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            find_transform(test_coo1, test_coo2)
