# Aafitrans (AstroAlign FInd TRANSform)

Aafitrans is a Python package that provides a modified version of the `find_transform` function from the Astroalign package. The modifications made to the function are as follows:

* The `sep` and `bottleneck` packages are no longer required as dependencies.
* Offers greater flexibility by supporting all transformations available in the `skimage` library.
* The `seed` parameter can be set during each run to ensure reproducibility of results.
* The `kdtree_search_radius` parameter can be set to achieve better and faster matches.

I have only made minor modifications to the source code to achieve the above. All credit goes to the original Astroalign authors for developing such amazing software. 
  
## Please cite:
```
Astroalign: A Python module for astronomical image registration.
Beroiz, M., Cabral, J. B., & Sanchez, B.
Astronomy & Computing, Volume 32, July 2020, 100384.
```

