'''
MIT License

Copyright (c) 2016-2019 Martin Beroiz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified by Prajwel Joseph
'''

'''
The following paper should be cited if you use the script in a scientific
publication

Astroalign: A Python module for astronomical image registration.
Beroiz, M., Cabral, J. B., & Sanchez, B.
Astronomy & Computing, Volume 32, July 2020, 100384.
'''


import numpy as np
from scipy.spatial import KDTree
from itertools import combinations
from functools import partial
from collections import Counter
from skimage import transform


__version__ = '0.1.0'

class _MatchTransform:
    def __init__(self, source, target, ttype):
        self.source = source
        self.target = target
        self.ttype = ttype

    def fit(self, data):
        """
        Return the best 2D similarity transform from the points given in data.

        data: N sets of similar corresponding triangles.
            3 indices for a triangle in ref
            and the 3 indices for the corresponding triangle in target;
            arranged in a (N, 3, 2) array.
        """
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        approx_t = transform.estimate_transform(
            self.ttype, self.source[s], self.target[d]
        )
        return approx_t

    def get_error(self, data, approx_t):
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        resid = approx_t.residuals(self.source[s], self.target[d]).reshape(
            d1, d2
        )
        error = resid.max(axis=1)
        return error
    
    def get_total_error(self, data, approx_t):
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        resid = approx_t.residuals(self.source[s], self.target[d])
        total_error = np.sqrt(np.sum(np.square(resid)) / (len(resid) * 3))    
        return total_error       
    
    
def _invariantfeatures(x1, x2, x3):
    """Given 3 points x1, x2, x3, return the invariant features for the set."""
    sides = np.sort(
        [
            np.linalg.norm(x1 - x2),
            np.linalg.norm(x2 - x3),
            np.linalg.norm(x1 - x3),
        ]
    )
    return [sides[2] / sides[1], sides[1] / sides[0]]    


def _arrangetriplet(sources, vertex_indices):
    """Order vertex_indices according to length side.

    Order in (a, b, c) form Where:
      a is the vertex defined by L1 & L2
      b is the vertex defined by L2 & L3
      c is the vertex defined by L3 & L1
    and L1 < L2 < L3 are the sides of the triangle
    defined by vertex_indices.
    """
    ind1, ind2, ind3 = vertex_indices
    x1, x2, x3 = sources[vertex_indices]

    side_ind = np.array([(ind1, ind2), (ind2, ind3), (ind3, ind1)])
    side_lengths = list(map(np.linalg.norm, (x1 - x2, x2 - x3, x3 - x1)))
    l1_ind, l2_ind, l3_ind = np.argsort(side_lengths)

    # the most common vertex in the list of vertices for two sides is the
    # point at which they meet.

    count = Counter(side_ind[[l1_ind, l2_ind]].flatten())
    a = count.most_common(1)[0][0]
    count = Counter(side_ind[[l2_ind, l3_ind]].flatten())
    b = count.most_common(1)[0][0]
    count = Counter(side_ind[[l3_ind, l1_ind]].flatten())
    c = count.most_common(1)[0][0]

    return np.array([a, b, c])


def _generate_invariants(sources, num_nearest_neighbors):
    """Return an array of (unique) invariants derived from the array `sources`.

    Return an array of the indices of `sources` that correspond to each
    invariant, arranged as described in _arrangetriplet.
    """


    arrange = partial(_arrangetriplet, sources=sources)

    inv = []
    triang_vrtx = []
    coordtree = KDTree(sources)
    # The number of nearest neighbors to request (to work with few sources)
    knn = min(len(sources), num_nearest_neighbors)
    for asrc in sources:
        __, indx = coordtree.query(asrc, knn)

        # Generate all possible triangles with the 5 indx provided, and store
        # them with the order (a, b, c) defined in _arrangetriplet
        all_asterism_triang = [
            arrange(vertex_indices=list(cmb)) for cmb in combinations(indx, 3)
        ]
        triang_vrtx.extend(all_asterism_triang)

        inv.extend(
            [
                _invariantfeatures(*sources[triplet])
                for triplet in all_asterism_triang
            ]
        )

    # Remove here all possible duplicate triangles
    uniq_ind = [
        pos for (pos, elem) in enumerate(inv) if elem not in inv[pos + 1 :]
    ]
    inv_uniq = np.array(inv)[uniq_ind]
    triang_vrtx_uniq = np.array(triang_vrtx)[uniq_ind]

    return inv_uniq, triang_vrtx_uniq


def find_transform(source, target, 
                   max_control_points=50, 
                   ttype='similarity', 
                   pixel_tolerance=2, 
                   min_matches=4, 
                   num_nearest_neighbors=8,
                   kdtree_search_radius = 0.02,
                   n_samples = 1,
                   get_best_fit = True,
                   seed = None):
    """Estimate the transform between ``source`` and ``target``.

    Return a GeometricTransform object ``T`` that maps pixel x, y indices from
    the source image s = (x, y) into the target (destination) image t = (x, y).
    T contains parameters of the tranformation.

    Parameters
    ----------
        source
            An iterable of (x, y) coordinates of the source control points.
        target
            An iterable of (x, y) coordinates of the target control points.
        max_control_points
            The maximum number of control point-sources to find the transformation.
        ttype
            The type of Transform to be estimated.
        pixel_tolerance
            The maximum residual error for the estimated tranform.            
        min_matches
            The minimum number of matches to be found.
        num_nearest_neighbors
            The number of nearest neighbors of a given star (including itself) 
            to construct the triangle invariants.                      
        kdtree_search_radius
            The default is 0.02. This radius is used to find nearest neighbours
            while conducting a KD tree search of invariant features. 
        n_samples
            The minimum number of data points to fit the model to. A value of 1
            refers to 1 triangle, corresponding to 3 pairs of coordinates. 
        get_best_fit
            Whether to minimise the total error.                          
        seed
            Seed value for Numpy Random Generator.       
            
    Returns
    -------
        T, (source_pos_array, target_pos_array)
            The transformation object and a tuple of corresponding star positions
            in source and target.

    Raises
    ------
        TypeError
            If input type of ``source`` or ``target`` is not supported.
        ValueError
            If it cannot find more than 3 stars on any input.
        MaxIterError
            If no transformation is found.
    """
    
    try:
        source_controlp = np.array(source)[:max_control_points]
        target_controlp = np.array(target)[:max_control_points]
    except Exception:
        raise TypeError("Input type for source not supported.")

    # Check for low number of reference points
    if len(source_controlp) < 3:
        raise ValueError(
            "Reference stars in source image are less than the "
            "minimum value (3)."
        )
    if len(target_controlp) < 3:
        raise ValueError(
            "Reference stars in target image are less than the "
            "minimum value (3)."
        )

    source_invariants, source_asterisms = _generate_invariants(source_controlp, num_nearest_neighbors)
    source_invariant_tree = KDTree(source_invariants)

    target_invariants, target_asterisms = _generate_invariants(target_controlp, num_nearest_neighbors)
    target_invariant_tree = KDTree(target_invariants)

    # r = 0.1 is the maximum search distance, 0.1 is an empirical value that
    # returns about the same number of matches than inputs
    # matches_list is a list of lists such that for each element
    # source_invariant_tree.data[i], matches_list[i] is a list of the indices
    # of its neighbors in target_invariant_tree.data
    matches_list = source_invariant_tree.query_ball_tree(
        target_invariant_tree, r=kdtree_search_radius
    )

    # matches unravels the previous list of matches into pairs of source and
    # target control point matches.
    # matches is a (N, 3, 2) array. N sets of similar corresponding triangles.
    # 3 indices for a triangle in ref
    # and the 3 indices for the corresponding triangle in target;
    matches = []
    # t1 is an asterism in source, t2 in target
    for t1, t2_list in zip(source_asterisms, matches_list):
        for t2 in target_asterisms[t2_list]:
            matches.append(list(zip(t1, t2)))
    matches = np.array(matches)

    inv_model = _MatchTransform(source_controlp, target_controlp, ttype)
    # Set the minimum matches to be between 1 and 10 asterisms
    # min_matches = max(1, min(10, int(min_matches)))

    if (len(source_controlp) == 3 or len(target_controlp) == 3) and len(
        matches
    ) == 1:
        best_t = inv_model.fit(matches)
        inlier_ind = np.arange(len(matches))  # All of the indices
    else:
        best_t, inlier_ind = _ransac(matches, 
                                     inv_model, 
                                     pixel_tolerance, 
                                     min_matches, 
                                     n_samples, 
                                     get_best_fit,
                                     seed)
        
    triangle_inliers = matches[inlier_ind]
    d1, d2, d3 = triangle_inliers.shape
    inl_arr = triangle_inliers.reshape(d1 * d2, d3)
    inl_arr_unique = np.unique(inl_arr, axis=0)
    s, t = inl_arr_unique.T

    return best_t, (source_controlp[s], target_controlp[t])



# Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.

#     * Neither the name of the Andrew D. Straw nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# a PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Modified by Martin Beroiz
# Modified by Prajwel Joseph


class MaxIterError(RuntimeError):
    """Raise if maximum iterations reached."""
    pass


def _ransac(data, model, thresh, min_matches, n_samples = 1, get_best_fit = True, seed = None):
    """Fit model parameters to data using the RANSAC algorithm.

    Parameters
    ----------
        data
            A set of data points
        model
            A model that can be fitted to data points
        thresh
            A threshold value to determine when a data point fits a model
        min_matches
            The min number of matches required to assert that a model
            fits well to data
        n_samples
            The minimum number of data points to fit the model to.
        get_best_fit
            Whether to minimise the total error.            
        seed
            Seed value for Numpy Random Generator.     
    Returns
    -------
        bestfit: model parameters which best fit the data (or nil if no good
                  model is found)
    """
    good_fit = None
    n_data = data.shape[0]
    all_idxs = np.arange(n_data)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_idxs)
    best_error = np.inf
    improve_error_counter = 0
    
    for iter_i in range(n_data):
        # Partition indices into two random subsets
        maybe_idxs = all_idxs[iter_i : iter_i + n_samples]
        test_idxs = np.concatenate([all_idxs[:iter_i], 
                                    all_idxs[iter_i + n_samples:]])        
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < thresh]
        alsoinliers = data[also_idxs, :]
        if len(alsoinliers) >= min_matches:
            good_data = np.concatenate((maybeinliers, alsoinliers))
            good_fit = model.fit(good_data)
            total_error = model.get_total_error(good_data, good_fit)
            if get_best_fit and total_error < best_error:
                best_error = total_error
                best_fit = good_fit 
                improve_error_counter += 1
                if improve_error_counter == 100:
                    break
            else:
                best_fit = good_fit

    if best_fit is None:
        raise MaxIterError(
            "List of matching triangles exhausted before an acceptable "
            "transformation was found"
        )

    better_fit = best_fit
    previous_fit = best_fit.params + 1
    for i in range(100):
        test_err = model.get_error(data, better_fit)
        better_inlier_idxs = np.arange(n_data)[test_err < thresh]
        better_data = data[better_inlier_idxs]
        better_fit = model.fit(better_data)
        if np.all((previous_fit == better_fit.params)):
            break
        previous_fit = better_fit.params

    return better_fit, better_inlier_idxs
