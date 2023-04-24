# Aafitrans (AstroAlign FInd TRANSform)

Aafitrans is a Python package that provides a modified version of the `find_transform` function from the Astroalign package. The modifications made to the function are as follows:

* The `sep` and `bottleneck` packages are no longer required as dependencies.
* Supports all transformations available in the `skimage` library.
* The `seed` parameter can be set during each run to ensure the reproducibility of results.
* The `kdtree_search_radius` parameter can be set to achieve faster matches.
* The RANSAC algorithm was modified so that the obtained solution corresponds to the one with the minimum sum of squared residuals.

Please see the original Astroalign software at https://github.com/quatrope/astroalign

  
## Please cite:
```
Astroalign: A Python module for astronomical image registration.
Beroiz, M., Cabral, J. B., & Sanchez, B.
Astronomy & Computing, Volume 32, July 2020, 100384.
```

