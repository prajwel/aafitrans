# Aafitrans (AstroAlign FInd TRANSform)

Aafitrans is a Python package that provides a modified version of the `find_transform` function from the Astroalign package. The modifications made to the function are as follows:

* The RANSAC algorithm was modified so that the obtained solution corresponds to the one with the minimum sum of squared residuals.
* Supports all transformations available in the `skimage` library.
* The `kdtree_search_radius` parameter can be set to achieve faster matches.
* The `seed` parameter can be set during each run to ensure the reproducibility of results.
* The `sep` and `bottleneck` packages are no longer required as dependencies.


Please see the original Astroalign software at https://github.com/quatrope/astroalign

  
## Citation:
```
Astroalign: A Python module for astronomical image registration.
Beroiz, M., Cabral, J. B., & Sanchez, B.
Astronomy & Computing, Volume 32, July 2020, 100384.
```


## Installation
```bash
pip install aafitrans
```

## Usage 
It is similar to Astroalign's [`find_transform`](https://astroalign.quatrope.org/en/latest/tutorial.html#finding-the-transformation) function. However, there are many parameters available for the user to modify. 
```python
from aafitrans import find_transform
transf, (matched_source_xy, matched_target_xy) = find_transform(source_xy, target_xy,
                                                                max_control_points=50,
                                                                ttype='similarity',
                                                                pixel_tolerance=2,
                                                                min_matches=4,
                                                                num_nearest_neighbors=8,
                                                                kdtree_search_radius=0.02,
                                                                n_samples=1,
                                                                get_best_fit=True,
                                                                seed=None)

```

## Documentation for `find_transform` function

The `find_transform` function estimates the transform between two sets of control points, source, and target. It returns a GeometricTransform object `T` ([see scikit-image documenation for details](https://scikit-image.org/docs/stable/auto_examples/transform/plot_transform_types.html#sphx-glr-auto-examples-transform-plot-transform-types-py)) that maps pixel x, y indices from the source image s = (x, y) into the target (destination) image t = (x, y). 

### Parameters:
- `source`: An iterable of (x, y) coordinates of the source control points.
- `target`: An iterable of (x, y) coordinates of the target control points.
- `max_control_points`: Default value is 50. The maximum number of control point-sources to find the transformation.
- `ttype`: Default value is `similarity`. The type of Transform to be estimated. One of the following should be set: {‘euclidean’, similarity’, ‘affine’, ‘piecewise-affine’, ‘projective’, ‘polynomial’}. For details, see [scikit-image documentation](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.estimate_transform). 
- `pixel_tolerance`: The maximum residual error for the estimated tranform.            
- `min_matches`: The minimum number of matches to be found. A value of 1 refers to 1 triangle, corresponding to 3 pairs of coordinates. 
- `num_nearest_neighbors`: The number of nearest neighbors of a given star (including itself) to construct the triangle invariants.                      
- `kdtree_search_radius`: The default is 0.02. This radius is used to find nearest neighbours while conducting a KD tree search of invariant features. 
- `n_samples`: The minimum number of data points to fit the model to. A value of 1 refers to 1 triangle, corresponding to 3 pairs of coordinates. 
- `get_best_fit`: Whether to minimize the total error.                          
- `seed`: Seed value for Numpy Random Generator.

### Returns:
- `T`: GeometricTransform object that maps pixel x, y indices from the source image s = (x, y) into the target (destination) image t = (x, y). It contains parameters of the transformation.
- `(source_pos_array, target_pos_array)`: A tuple of corresponding star positions in source and target.

### Raises:
- `TypeError`: If input type of `source` or `target` is not supported.
- `ValueError`: If it cannot find more than 3 stars on any input.
- `MaxIterError`: If no transformation is found.
