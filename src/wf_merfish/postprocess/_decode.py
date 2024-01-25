"""
Functions to decode barcoded FISH experiments.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import cdist
from scipy.special import binom
import itertools
import functools
import operator
import napari
from tqdm import tqdm
from numpy.random import default_rng
import matplotlib.lines as mlines
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from numbers import Number
from sklearn.neighbors import BallTree

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TO DO: - tear out all functions that are not used in decoding.
#          furthermore, need to remove old decoding method.
#          simulation code can live in it's own repo, this should be a clean
#          file just for decoding.
#        - type hints and docstrings for all functions 
#        - what is the plan with the stepwise decoding?
#        - how to rework for a better dask strategy?

def make_mask_from_used_spots(n_spots, combinations, select, reverse=False):
    """
    Make a boolean mask to filter spots used in combinations.
    
    Example
    -------
    >>> combinations = optim_results['spots_combinations']
    >>> select = optim_results['best_combi']
    >>> mask = mask_from_used_spots(len(coords), combinations, select)
    """
    
    # flatten the list of tuples
    used_spots = list(itertools.compress(combinations, select))
    # get unique spot ids
    uniq_spots = unique_nested_iterable(used_spots)
    # build the bolean mask
    select = np.full(n_spots, True)
    select[uniq_spots] = False
    
    if reverse:
        return ~select
    return select


def compute_distances(
    source, target, dist_method="xy_z_orthog", metric="euclidean", tilt_vector=None
):
    """
    Parameters
    ----------
    source : ndarray
        Coordinates of the first set of points.
    target : ndarray
        Coordinates of the second set of points.
    dist_method : str
        Method used to compute distances. If 'isotropic', standard distances are computed considering all axes
        simultaneously. If 'xy_z_orthog' 2 distances are computed, for the xy plane and along the z axis
        respectively. If 'xy_z_tilted' 2 distances are computed for the tilted plane and its normal axis.

    Example
    -------
    >>> source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> target = np.array([[0, 0, 0], [-3, 0, 2], [0, 0, 10]])
    >>> compute_distances(source, target)
        (array([0, 4, 0]), array([0., 2., 5.]))
    >>> compute_distances(source, target, metric='L1')
        (array([0, 4, 0]), array([0, 2, 7]))

    """
    if dist_method == "isotropic":
        dist = cdist(source, target, metric=metric)
        return dist

    elif dist_method == "xy_z_orthog":
        dist_xy = cdist(source[:, 1:], target[:, 1:], metric=metric)
        dist_z = cdist(
            source[:, 0].reshape(-1, 1), target[:, 0].reshape(-1, 1), metric=metric
        )
        return dist_z, dist_xy

    elif dist_method == "xy_z_tilted":
        raise NotImplementedError("Method 'xy_z_tilted' will be implemented soon")


def find_neighbor_spots_in_round(
    source,
    target,
    dist_params,
    dist_method="isotropic",
    metric="euclidean",
    return_bool=False,
):
    """
    For each spot in a given round ("source"), find if there are neighbors
    in another round ("target") within a given distance.

    Parameters
    ----------
    source : ndarray
        Coordinates of spots in the source round.
    target : ndarray
        Coordinates of spots in the target round.
    dist_method : str
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog
    dist_params : float or array
        Threshold distance to classify spots as neighbors.
        Multiple threshold can be used depending on the method, typically 2
        to have a threshold for the xy plane and one for the z axis.
    return_bool : bool
        If True, return a vector indicating the presence of neighbors
        for spots in the source set.

    Returns
    -------
    pairs : ndarray
        Pairs of neighbors.
    has_neighb : array
        Array indicating the presence of neighbors for each spot in their source round.

    Example
    -------
    >>> source = np.array([[0, 0, 0],
                           [0, 2, 0]])
    >>> target = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 2, 0],
                           [0, 0, 3]])
    """

    # Compute all distances between spots of given round and all other spots of other round
    dist = compute_distances(source, target, dist_method=dist_method, metric=metric)
    # check if distances below threshold for all dimensions
    if dist_method == "xy_z_orthog":
        is_neighb = np.logical_and(dist[0] < dist_params[0], dist[1] < dist_params[1])
    elif dist_method == "isotropic":
        is_neighb = dist < dist_params

    if return_bool:
        # detect if there is any neighbor for each spot
        has_neighb = np.any(is_neighb, axis=1)
        return has_neighb
    else:
        # extract pairs of neighboring spots
        y, x = np.where(is_neighb)
        pairs = np.vstack([y, x]).T
        return pairs


def make_all_rounds_pairs(start=0, end=16):
    pairs_rounds = list(itertools.permutations(range(start, end), 2))
    return pairs_rounds


def assemble_barcodes(neighbors, fill_value=None):
    """
    Parameters
    ----------
    neighbors : dict[dict[array]]
        Dictionnary of dictionnaries, where the fist level of keys is the
        set of source rounds, and the second level of key is the set of
        target round. Each second level value is an array indicating the
        presence of neighbors from spots in the source round to spots in
        the target round.

    Returns
    -------
    barcodes : dict[array]
        Dictionnary of barcodes found for each spot in a source round,
        keys indicate the id of the source round.

    Example
    -------
    >>> neighbors = {2: {1: [0, 1, 2, 3],
                         0: [4, 5, 6, 7]},
                     0: {1: [8, 9, 10, 11],
                         2: [12, 13, 14, 15]},
                     1: {2: [16, 17, 18, 19],
                         0: [20, 21, 22, 23]}}
    >>> assemble_barcodes(neighbors)
    {2: array([[4, 0, 1],
               [5, 1, 1],
               [6, 2, 1],
               [7, 3, 1]]),
     0: array([[ 1,  8, 12],
               [ 1,  9, 13],
               [ 1, 10, 14],
               [ 1, 11, 15]]),
     1: array([[20,  1, 16],
               [21,  1, 17],
               [22,  1, 18],
               [23,  1, 19]])}
    """

    # dictionary storing all barcodes matrices for each round
    barcodes = {}
    # for each round, stack vectors of neighbors into arrays across target rounds
    for round_source, round_targets in neighbors.items():
        # get sorted list of target round IDs
        round_ids = np.unique([i for i in round_targets.keys()])
        # initialize empty array
        nb_neigh = len(round_targets[round_ids[0]])
        nb_rounds = round_ids.size + 1  # because we consider current source round
        # get the type of data and choose between1, 1.0 and True
        if fill_value is None:
            for round_id in round_ids:
                if len(round_targets[round_id]) > 0:
                    fill_value = round_targets[round_id][0]
                    break
            if isinstance(fill_value, bool):
                fill_value = True
            elif isinstance(fill_value, int):
                fill_value = 1
            else:
                fill_value = 1.0
        # initilize array, which sets bits of the current source round to 1 or True
        round_barcode = np.full(shape=(nb_neigh, nb_rounds), fill_value=fill_value)
        # stack each vector in the array
        for round_id in round_ids:
            round_barcode[:, round_id] = round_targets[round_id]
        # save the array in the barcode dictionary
        barcodes[round_source] = round_barcode
    return barcodes


def clean_barcodes(barcodes, coords, min=3, max=5):
    """
    Remove barcodes and their corresponding coordinates if they
    have too few or too many positive bits.

    Parameters
    ----------
    barcodes : dict[array]
        Dictionnary of barcodes found for each spot in a source round,
        keys indicate the id of the source round.
    coords : list(arrays)
        Coordinates of all spots in rounds.
    min : int
        Minimum number of positive bits each barcode should have.
    max : int
        Maximum number of positive bits each barcode should have.

    Returns
    -------
    barcodes : dict[array]
        Dictionnary of barcodes found for each spot in a source round,
        keys indicate the id of the source round.
    coords : list(arrays)
        Coordinates of all spots in rounds.

    Example
    -------
    >>> barcodes = {0: np.array([[1, 1, 1, 1, 1, 1],
                                 [0, 0, 1, 1, 1, 0]]),
                    1: np.array([[1, 0, 0, 1, 0, 0],
                                 [1, 0, 1, 0, 1, 1]])}
    >>> coords = [np.array([[1, 2, 3],
                            [4, 5, 6]]),
                  np.array([[7, 8, 9],
                            [10, 11, 12]])]
    >>> clean_barcodes(barcodes, coords, min=3, max=5)
    ({0: array([[0, 0, 1, 1, 1, 0]]), 1: array([[1, 0, 1, 0, 1, 1]])},
    [array([[4, 5, 6]]), array([[10, 11, 12]])])
    """

    for rd_id, rd_barcode in barcodes.items():
        select = np.logical_and(
            rd_barcode.sum(axis=1) >= min, rd_barcode.sum(axis=1) <= max
        )
        # selection by key to be sure assignment is effective, necessary?
        barcodes[rd_id] = rd_barcode[select, :]
        coords[rd_id] = coords[rd_id][select, :]
    return barcodes, coords


def merge_barcodes_pairs_rounds(
    barcodes_1,
    barcodes_2,
    coords_1,
    coords_2,
    dist_params,
    dist_method="isotropic",
    metric="euclidean",
):
    """
    Merge barcodes and their corresponding coordinates in a pair of rounds
    when they are identical and close enough to each other.

    Parameters
    ----------
    barcodes_1 : ndarray
        Barcodes of first round, shape (n_barcodes, n_rounds).
    barcodes_2 : ndarray
        Barcodes of second round, shape (n_barcodes, n_rounds).
    coords_1 : ndarray
        Coordinates of barcodes of first round, shape (n_barcodes, dim_image).
    coords_2 : ndarray
        Coordinates of barcodes of second round, shape (n_barcodes, dim_image).
    dist_method : str
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog
    metric : str
        Metric used to compute distance between 2 points.
    dist_params : float or array
        Threshold distance to classify spots as neighbors.
        Multiple threshold can be used depending on the method, typically 2
        to have a threshold for the xy plane and one for the z axis.

    Returns
    -------
    barcodes_out : ndarray
        Merged barcodes.
    coords_out : ndarray
        Merged coordinates of barcodes.

    Example
    -------
    >>> barcodes_1 = np.array([[1, 1, 1, 1],
                               [0, 0, 0, 0]])
    >>> barcodes_2 = np.array([[1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
    >>> coords_1 = np.array([[0, 0, 0],
                             [1, 2, 2]])
    >>> coords_2 = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [2, 2, 2]])
    >>> merge_barcodes_pairs_rounds(barcodes_1, barcodes_2, coords_1, coords_2,
                                    dist_params=[0.6, 0.2])
        (array([[1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]]),
        array([[0, 0, 0],
                [1, 2, 2],
                [0, 0, 0],
                [2, 2, 2]]))
    """

    # find all pairs of neighbors between the 2 rounds
    pairs = find_neighbor_spots_in_round(
        coords_1,
        coords_2,
        dist_params=dist_params,
        dist_method=dist_method,
        metric=metric,
    )
    # change to dictionary to delete entries / indices without shifting
    # element by more than one index if previous elements need to be discarded
    barcodes_2 = array_to_dict(barcodes_2)
    coords_2 = array_to_dict(coords_2)

    # very manual iteration to allow modification of the `pairs` array
    # while iterating over it
    k = 0
    while k < len(pairs):
        i, j = pairs[k]
        if np.all(barcodes_1[i] == barcodes_2[j]):
            # delete barcode and coordinates in the second set
            del barcodes_2[j]
            del coords_2[j]
            select = np.logical_or(np.arange(len(pairs)) <= k, pairs[:, 1] != j)
            pairs = pairs[select, :]
        k += 1
    # convert back to array for stacking and future distance computation
    barcodes_2 = dict_to_array(barcodes_2)
    coords_2 = dict_to_array(coords_2)

    # stack all remaining barcodes and coordinates
    if len(barcodes_2) != 0:
        barcodes_out = np.vstack([barcodes_1, barcodes_2])
        coords_out = np.vstack([coords_1, coords_2])
    else:
        barcodes_out = barcodes_1
        coords_out = coords_1

    return barcodes_out, coords_out


def make_pyramidal_pairs(base):
    """
    Make successive lists resulting from merging pairs in previous list,
    until a list of a unique pair is reached.

    Parameters
    ----------
    base : list
        A list of elements that will be successively merged.

    Returns
    -------
    pyramid : list
        A list of lists, each of them containing pairs of merged
        items from the previous list.

    Example
    -------
    >>> base = list(range(5))
    >>> make_pyramidal_pairs(base)
    [[0, 1, 2, 3, 4], [[0, 1], [2, 3], [4]], [[0, 2], [4]], [[0, 4]]]
    """

    # Make first level of the pyramid with base
    pyramid = [base]
    # First iteration in numbers, not pairs of number
    level = [
        [pyramid[-1][2 * i], pyramid[-1][2 * i + 1]]
        for i in range(len(pyramid[-1]) // 2)
    ]
    if len(pyramid[-1]) % 2 == 1:
        level.append([pyramid[-1][-1]])
    pyramid.append(level)
    # Next iterations on pairs of numbers
    while len(pyramid[-1]) > 1:
        level = [
            [pyramid[-1][2 * i][0], pyramid[-1][2 * i + 1][0]]
            for i in range(len(pyramid[-1]) // 2)
        ]
        if len(pyramid[-1]) % 2 == 1:
            level.append([pyramid[-1][-1][0]])
        pyramid.append(level)
    return pyramid


def merge_barcodes(
    barcodes, coords, dist_params, dist_method="xy_z_orthog", metric="euclidean", verbose=0,
):
    """
    Merge all barcodes and their corresponding coordinates in all rounds
    when they are identical and close enough to each other.

    Parameters
    ----------
    barcodes : ndarray
        Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).
    coords : ndarray
        Coordinates of barcodes, shape (n_barcodes, dim_image).
    dist_method : str
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog
    metric : str
        Metric used to compute distance between 2 points.
    dist_params : float or array
        Threshold distance to classify spots as neighbors.
        Multiple threshold can be used depending on the method, typically 2
        to have a threshold for the xy plane and one for the z axis.

    Returns
    -------
    barcodes_out : ndarray
        Merged barcodes.
    coords_out : ndarray
        Merged coordinates of barcodes.

    Example
    -------
    >>> barcodes = {0: np.array([[1, 1, 1, 1],
                                 [1, 1, 0, 0],
                                 [1, 0, 1, 0],
                                 [1, 0, 0, 1]]),
            1: np.array([[1, 1, 1, 1],
                         [0, 1, 1, 0]]),
            2: np.array([[1, 1, 1, 1],
                         [1, 0, 1, 0]]),
            3: np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 0, 0, 1]])}
    >>> coords = [np.zeros((len(barcodes[0]), 3), dtype='int'),
                  np.zeros((len(barcodes[1]), 3), dtype='int'),
                  np.zeros((len(barcodes[2]), 3), dtype='int'),
                  np.array([[0, 0, 0],
                            [5, 5, 5],
                            [0, 0, 0]], dtype='int')]
    >>> merge_barcodes(barcodes, coords, dist_params=[0.6, 0.2])
    """

    # get sorted list of round IDs
    round_ids = np.unique([i for i in barcodes.keys()])

    # Pyramidal merge of pairs of rounds
    pyram_levels = make_pyramidal_pairs(round_ids)
    # something like [[0, 1, 2, 3], [[0, 1], [2, 3]], [[0, 2]]]
    for level_pairs in pyram_levels[1:]:
        # for ex: [[0, 1], [2, 3]]
        for pair in level_pairs:
            # for ex: [0, 1]
            if len(pair) == 2:
                # avoid runing on a singlet
                barcodes_1 = barcodes[pair[0]]
                barcodes_2 = barcodes[pair[1]]
                coords_1 = coords[pair[0]]
                coords_2 = coords[pair[1]]
                
                if verbose > 0:
                    print("pair:", pair)
                if verbose > 1:
                    print("barcodes_1 shape:", barcodes_1.shape)
                    print("barcodes_2 shape:", barcodes_2.shape)
                    print("barcodes", barcodes)
                    print("coords", coords)
                barcodes[pair[0]], coords[pair[0]] = merge_barcodes_pairs_rounds(
                    barcodes_1,
                    barcodes_2,
                    coords_1,
                    coords_2,
                    dist_params=dist_params,
                    dist_method=dist_method,
                    metric=metric,
                )
                # # clean-up space
                barcodes[pair[1]] = None
                coords[pair[1]] = None
    return barcodes[0], coords[0]


def correct_barcode(barcode, codebook_keys, codebook_vals, max_dist=1):
    """
    Infer the identity of a barcode that has some errors from a codebook.

    Parameters
    ----------
    barcodes : ndarray
        Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).
    codebook_keys : array
        List of keys of the codebook.
    codebook_vals : ndarray
        2D array of binary values of the codebook.

    Example
    -------
    barcode = np.array([1,1,1,0,0,0,0,0])
    codebook = {'a': np.array([0,0,0,0,1,1,1,1]),
                'b': np.array([1,1,1,1,0,0,0,0]),
                'c': np.array([1,0,1,0,1,0,1,0])}
    cbk_keys, cbk_vals = dict_to_2D_array(codebook)

    """
    # compute distances between barcode and codebook's barcodes
    dist = cdist(barcode.reshape(1, -1), codebook_vals, metric='cityblock')
    # detect the one within an acceptable distance
    select_ids = np.where(dist <= max_dist)[1]
    # infer identity
    if len(select_ids) == 0:
        bcd_species = None
    else:
        bcd_species = codebook_keys[select_ids[0]]
    return bcd_species


def infer_species_from_barcodes(barcodes, codebook, method='deterministic',
                                err_max_dist=1):
    """
    Guess the species identities of spots from their barcodes and a codebook.

    Parameters
    ----------
    barcodes : ndarray
        Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).
    codebook : dict
        Association from species to rounds.
    method : str
        Method used for inference. Default is 'deterministic', where a single estimate is
        made for exactly matches between bacordes and the codebook. A bayesian method will
        be implemented soon.
    err_max_dist : int
        Maximum distance between a measured barcode and it's real barcode sequence. 
    """

    if method == 'deterministic':
        # make dictionnary barcodes --> species from dictionnary species --> barcodes
        inv_codebook = {val: key for key, val in codebook.items()}
        species = []
        # make arrays of keys and values of the codebook for error correction step
        cbk_keys, cbk_vals = dict_to_2D_array(codebook)
        cbk_vals = array_str_to_int(cbk_vals)
        # decode spot identities
        for barcode in barcodes:
            # transform numerical barcodes to string ones
            barcode_str = ''.join([str(int(i)) for i in barcode])
            # actual decoding
            if barcode_str in inv_codebook.keys():
                bcd_species = inv_codebook[barcode_str]
            else:
                # Error correction code
                bcd_species = correct_barcode(barcode, cbk_keys, cbk_vals, max_dist=err_max_dist)
            species.append(bcd_species)

    elif method == 'bayesian':
        raise NotImplementedError("Method 'bayesian' will be implemented soon")

    return species, barcodes

def decode_spots(coords, codebook, dist_params, dist_method="isotropic", metric="euclidean",
                 clean_min=3, clean_max=5, verbose=1):
    """
    For each spot in a given round, find if there are neighbors
    in each other rounds within a given distance, and reconstruct barcodes from that.

    Parameters
    ----------
    coords : list(arrays)
        Coordinates of all spots in rounds.
    codebook : dict
        Association from species to rounds.
    dist_params : float or array
        Threshold distance to classify spots as neighbors.
        Multiple threshold can be used depending on the method, typically 2
        to have a threshold for the xy plane and one for the z axis.
    dist_method : str
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog
    metric : str
        Metric used to compute distance between 2 points.

    Returns
    -------
    species : list
        Species of decoded spots.
    coords : array
        Coordinates of decoded spots.
    barcode : array
        Reconstructed barcode from data around each spot location.

    """

    if verbose > 0:
        print("Making round pairs")
    coords = deepcopy(coords)
    nb_rounds = len(coords)
    if isinstance(dist_params, Number):
        dist_method = 'isotropic'
    round_pairs = make_all_rounds_pairs(start=0, end=nb_rounds)

    # store all potential neighbors decected from each round to the other
    if verbose > 0:
        print("Finding neighbors across rounds")
    neighbors = {i: {} for i in range(nb_rounds)}
    for pair in round_pairs:
        if verbose > 1:
            print("    rounds pair:", pair)
        neighbors[pair[0]][pair[1]] = find_neighbor_spots_in_round(
            coords[pair[0]],
            coords[pair[1]],
            dist_method=dist_method,
            metric=metric,
            dist_params=dist_params,
            return_bool=True,
        )
    if verbose > 0:
        print("Assembling barcodes")
    barcodes = assemble_barcodes(neighbors, fill_value=True)
    # remove barcodes that have too few or too many positive bits
    if verbose > 0:
        print("Cleaning barcodes")
    barcodes, coords = clean_barcodes(barcodes, coords, min=clean_min, max=clean_max)

    if verbose > 0:
        print("Merging barcodes")
    barcodes, coords = merge_barcodes(
        barcodes,
        coords,
        dist_params=dist_params,
        dist_method=dist_method,
        metric=metric,
    )

    if verbose > 0:
        print("Infering species")
    species, barcodes = infer_species_from_barcodes(barcodes, codebook)

    if verbose > 0:
        print("Spot decoding done")
    return species, coords, barcodes


def plot_compare_decoded_spots(true_coords=None, true_species=None, 
                               decoded_coords=None, decoded_species=None, 
                               cmap=None, figsize=(10, 10), plot_legend=True):
    """
    Plot ground truth and decoded species and coordinates of spots.

    Parameters
    ----------
    true_coords : array or dataframe
        Ground truth coordinates of spots. 
    true_species : list or array
        Ground truth species of FISH spots.
    decoded_coords : array or dataframe
        Output coordinates (merged and cleaned) from the spot decoding pipeline.
    decoded_species : list or array
        Output species from the spot decoding pipeline.
    cmap : dict
        Association between species and colors, default is None.
    figsize : tuple or array
        Size of the scatter plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    uniq_spec = set.union(set(np.unique(true_species).tolist()), 
                        set(np.unique(decoded_species).tolist()))

    if cmap is None:
        palette = [mpl.colors.rgb2hex(x) for x in mpl.cm.get_cmap('tab20').colors]
        n_colors = len(palette)
        cmap = {x: palette[i % n_colors] for i, x in enumerate(uniq_spec)}

    artists = []
    if true_coords is not None:
        if isinstance(true_species, (list, pd.Series)):
            true_species = np.array(true_species)
        if isinstance(true_coords, pd.DataFrame):
            true_coords = np.array(true_coords)
        for spec in uniq_spec:
            select = true_species == spec
            plot_coords = true_coords[select, -2:]
            artist_true  = ax.scatter(plot_coords[:, 1], plot_coords[:, 0], label=f'true {spec}', marker='o', s=50, edgecolors=cmap[spec], facecolors='none')
            if decoded_coords is None:
                artists.append(artist_true)
            
    if decoded_coords is not None:
        if true_coords is None:
            str_decoded = 'decoded '
        else:
            str_decoded = ''
        if isinstance(decoded_species, (list, pd.Series)):
            decoded_species = np.array(decoded_species)
        if isinstance(decoded_coords, pd.DataFrame):
            decoded_coords = np.array(decoded_coords)
        for spec in uniq_spec:
            select = decoded_species == spec
            # maybe we will need to handle species in true set but not in decoded one
            plot_coords = decoded_coords[select, -2:]
            artist_decoded = ax.scatter(plot_coords[:, 1], plot_coords[:, 0], label=f'{str_decoded}{spec}', marker='.', c=cmap[spec])
            artists.append(artist_decoded)
    
    if plot_legend:
        if true_coords is not None and decoded_coords is not None:
            marker_decoded = mlines.Line2D([], [], color='black', marker='.', 
                                        linestyle='None',markersize=5, 
                                        label='Decoded species')
            artists.insert(0, marker_decoded)
            
            marker_true = mlines.Line2D([], [], color='black', markerfacecolor='none', 
                                        marker='o', linestyle='None',
                                        markersize=5, label='True species')
            artists.insert(0, marker_true)
        plt.legend(
            handles=artists,
            bbox_to_anchor=(1,1), 
            loc="upper left",
            )
        
    return fig, ax


def view_spot_images(
    species,
    codebook,
    all_rounds_images=None,
    true_species=None, 
    true_coords=None,
    decoded_species=None, 
    decoded_coords=None,
    localized_coords=None,
    viewer=None,
    scale_img=None,
    scale_pts=None,
    ):
    """
    Display detected species and the images it appears in.
    """

    bcd_sequence = codebook[species]
    # go from '01001101' to [1, 4, 5, 7]
    spec_rounds = [i for i, x in enumerate(bcd_sequence) if x == '1']

    return_viewer = False
    if viewer is None:
        viewer = napari.Viewer()
        return_viewer = True
    
    # add images where spots of the given should appear
    if all_rounds_images is not None:
        for r_idx in spec_rounds:
            viewer.add_image(
                all_rounds_images[r_idx], 
                name=f"image {r_idx}", 
                scale=scale_img, 
                interpolation2d='nearest', 
                interpolation3d='nearest',
                )
    # add true spots' positions
    if true_species is not None:
        centers = true_coords[true_species == species]
        viewer.add_points(centers, name='true positions', size=.25, face_color='green', edge_color='black', scale=scale_pts)
    # add localized positions
    if localized_coords is not None:
        for r_idx in spec_rounds:
            centers = localized_coords.loc[localized_coords['round'] == r_idx, ['z', 'y', 'x']]
            viewer.add_points(centers, name=f'localized positions round {r_idx}', size=.25, face_color='yellow', edge_color='black', scale=scale_pts)
    # add decoded positions
    if decoded_species is not None:
        centers = decoded_coords[decoded_species == species]
        viewer.add_points(centers, name='decoded positions', size=.25, face_color='red', edge_color='black', scale=scale_pts)

    if return_viewer:
        return viewer


# -------------------------------------------------------------------------
# -------------------------- Major method rework --------------------------
# -------------------------------------------------------------------------

def explain():
    text = """
    The following functions aim for a more robust decoding pipeline with respect
    to non perfect image registration and spot localization. 
    The main idea is to implement a global optimization method to select barcodes
    among all possible ones when considering a larger distance threshold for 
    neighbor spots across images / rounds.
    *all possible* also means we build barcodes by switching off all combinations
    of positive bits, for example: 1101 --> 1001, 1000, 1100 etc...
    Another major difference with the previous functions is that we use localizations
    results, such as spots coordinates, amplitude, sigmas, etc... in the loss
    function during the optimization procedure.

    Spots have unique IDs, even across rounds.
    Pipeline:
    For each spot, detect all neighbors in other rounds.
    Then build all possible barcodes, while storing the contributing spots' IDs too.
    Barcodes can have identical sequences but being made of different spot_ids.
    Store in `barcodes` matrix / dataframe all possible barcodes, involved spot_ids
    and derived statistics from contributing spots (distance^2, mean_amplitude, ...).
    Filter out identical barcodes, i.e. identical sequences of contributing spots.
    Optimize selection of barcodes (easy to say!):
        Use different statistics stats := min dist^2, max number of decoded  
        molecules, min N corrected / n perfect sequences, min False Positive, ...
        alpha := vector of parameters weights
        L := loss function = alpha * stats
        --> standardize stats before multiplying with alpha
    Merge barcodes:
        This is traditionnaly performed around the end of the pipeline, but here
        it's part of the optimization phase: when a spot_id is used for a code_id,
        the sampler ignores all other code_ids related to this spot_id.
        This search can take some time, and is performed at all iterations of the
        optimization phase, that is why we implement a network representation of
        barcodes, in order to query quickly which ones are then allowed or forbidden.
    Deterministic decoding of sequences from the codebook.
    In the future, include for each spot its likelyhood of being a True Positive,
    and include it in the loss function, potentially for a bayesian method.

    More details:
    Building the network of barcodes: 
        nodes are barcodes' ids and edges are presence of a common contributing spot.
    We want to avoid computing the loss function over all barcodes at once, as
    there is some locality, and some clusters of nodes can reach an optimized state
    before the other ones, so we don't want to run computation on them again.
    So we optimize the loss functions L_i of each non-connected sub-network, or
    poorly connected sub-network ("cluster", "module", ...).
    The total loss function is the sum of all the sub-loss functions.
    For each cluster:
    Initialization:
        The sampler starts with codewords that have 4 positive bits and the minimal
        inner-spots distances.
    Iterations:
        Propose n_propose random code_ids (start with 1?).
        Switch off all connected barcodes.
        Normalize stats for L_i.
        Compute L_i, store in loss_gains, if it's the lowest store selection.
        If n_iter > n_iter_max or min(loss_gains[-patience:]) > tol, stop.
    """
    print(text)


def make_neighbors_lists(coords, spot_ids, spot_rounds, dist_params, max_positive_bits, 
                         dist_method="isotropic", verbose=1, **kwargs):
    """
    Build a list of neighbors' ids for each round, starting from each spot.

    Example
    --------
    >>> coords_r0 = np.array([[0, 0, 0],
                              [0, 2, 0]])
    >>> coords_r1 = np.array([[0, 0, 0],
                              [2, 0, 0],
                              [0, 2, 0],
                              [0, 2, 0],
                              [0, 0, 4]])
    >>> spot_rounds = np.array([0, 0, 1, 1, 1, 1, 1])
    >>> coords = np.vstack((coords_r0, coords_r1))
    >>> spot_ids = np.arange(len(coords))
    >>> make_neighbors_lists(coords, spot_ids, spot_rounds, dist_params=[1, 1])
    [((1,), (4, 5)),
     ((1,), (5,)),
     ((), (6,)),
     ((1,), (4,)),
     ((0,), (2,)),
     ((), (3,))]
    """

    uniq_rounds = np.unique(spot_rounds)
    neighbors = []
    if verbose > 0:
        iterable = tqdm(spot_ids, desc='spots ids')
    else:
        iterable = spot_ids
    # TODO: use BallTree between pairs of rounds
    for spot_id in iterable:
        r_id = spot_rounds[spot_id]
        spot_neighbs = []
        n_pos_bits = 0
        for r_trg in uniq_rounds:
            if r_trg == r_id:
                # add source spot's id
                spot_neighbs.append(tuple([spot_id]))
                n_pos_bits += 1
            else:
                select_trg = spot_rounds == r_trg
                source = coords[spot_id].reshape((1, -1))
                target = coords[select_trg, :]
                trg_ids = spot_ids[select_trg]
                pairs = find_neighbor_spots_in_round(
                            source,
                            target,
                            dist_params=dist_params,
                            dist_method=dist_method,
                            **kwargs,
                            )
                # save target neighbors indices
                # as a list as array don't work so well with make_combinations()
                bit_neighbs = tuple(trg_ids[pairs[:, 1]])
                if len(bit_neighbs) > 0:
                    n_pos_bits += 1
                spot_neighbs.append(bit_neighbs)
        if n_pos_bits < max_positive_bits:
            neighbors.append(tuple(spot_neighbs))
    # remove duplicate sets of neighbors
    neighbors = list(set(neighbors))
    
    return neighbors


def spots_product(*args, repeat=1, verbose=1):
    """
    Make combinations of elements, picking either a single element or none per list.
    Absence of element is indicated by -1. This is a modified receipe from itertools.

    Example
    -------
    >>> spots = [[1], [], [2, 3]]
    >>> list(spots_product(*spots))
        [[0, 0, 0], [0, 0, 2], [0, 0, 3], [1, 0, 0], [1, 0, 2], [1, 0, 3]]
    """
    
    pools = [pool for pool in args] * repeat
    result = [[]]
    
    if verbose > 1:
        iterable = tqdm(pools, desc='spots pool', leave=False)
    else:
        iterable = pools
    for pool in iterable:
        result = [x+[y] for x in result for y in [-1] + list(pool)]
    for prod in result:
        yield prod


def make_combinations_single_spot(spots, size_bcd_min=2, size_bcd_max=6):
    """
    Build barcodes from all possible combinations of localized spots' ids across rounds.

    Parameters
    ----------
    spots : list(list)
        Lists of neighbors in each rounds for each spot.

    Returns
    -------
    contribs : list(list)
        Contributing spots' ids of barcode sequences.
    
    Example
    -------
    >>> spots = [[2], [4, 5], [], [], [8]]
    >>> make_combinations_single_spot(spots, size_bcd_min=1)
    [[8],
    [4],
    [4, 8],
    [5],
    [5, 8],
    [2],
    [2, 8],
    [2, 4],
    [2, 4, 8],
    [2, 5],
    [2, 5, 8]]
    """

    # make all combinations of contributing spots
    products = spots_product(*spots)
    # remove 0s from list of contributing spots
    prod_cleaned = [[i for i in prod if i !=-1] for prod in products]
    del products
    # filter now by min and max number of contributing spots
    contribs = [tuple(i) for i in prod_cleaned if size_bcd_min <= len(i) <= size_bcd_max]

    return contribs


def make_combinations(neighbors, size_bcd_min=2, size_bcd_max=6, verbose=1):
    """
    
    Example
    -------
    >>> coords_r0 = np.array([[0, 0, 0],
                              [0, 2, 0]])
    >>> coords_r1 = np.array([[0, 0, 0],
                              [2, 0, 0],
                              [0, 2, 0],
                              [0, 3, 0],
                              [0, 0, 4]])
    >>> spot_rounds = np.array([0, 0, 1, 1, 1, 1, 1])
    >>> coords = np.vstack((coords_r0, coords_r1))
    >>> spot_ids = np.arange(len(coords))
    >>> uniq_rounds = np.unique(spot_rounds)
    >>> neighbors = make_neighbors_lists(coords, spot_ids, spot_rounds, dist_params=[1, 3])
    >>> make_combinations(neighbors, size_bcd_min=1, size_bcd_max=6)
    [[[2], [4], [0], [0, 2], [0, 4]],
    [[2], [4], [5], [1], [1, 2], [1, 4], [1, 5]],
    [[2], [0], [0, 2], [1], [1, 2]],
    [[3]],
    [[4], [0], [0, 4], [1], [1, 4]],
    [[5], [1], [1, 5]],
    [[6]]]
    """

    combinations = []
    if verbose > 0:
        iterable = tqdm(neighbors, desc='spots tuple')
    else:
        iterable = neighbors
    for spots in iterable:
        combinations.extend(
            make_combinations_single_spot(spots, size_bcd_min, size_bcd_max)
            )
        # uncomment to observe what's happening:
        # print('spots', spots, ' --> ', make_combinations_single_spot(spots, size_bcd_min, size_bcd_max))
    # remove duplicate sets of combinations
    combinations = list(set(combinations))

    return combinations


def make_barcode_from_contributors(contribs, spot_rounds, n_bits=None, verbose=1):
    """
    Parameters
    ----------
    contribs : list(list)
        Ids of spots contributing to barcode sequences.
    spot_rounds : array
        Round index of each spot.
    n_bits : int
        Number of bits in barcodes.

    Returns
    -------
    bcd_sequences : array
        Barcode sequences from all possible spots ids combinations.
    spots_bcd : dict
        Inform for each spot the barcode ids it contributes to. 

    Example
    -------
    >>> contribs = [(0, 5), (1, 3), (2, 4, 6)]
    >>> spot_rounds = np.array([0, 0, 1, 1, 2, 3, 3])
    >>> make_barcode_from_contributors(contribs, spot_rounds, n_bits=None)
    array([[1, 0, 0, 1],
           [1, 1, 0, 0],
           [0, 1, 1, 1]])
    """

    spots_bcd = {}
    if n_bits is None:
        n_bits = spot_rounds.max() + 1
    # make barcodes
    bcd_sequences = np.zeros((len(contribs), n_bits), dtype=int)
    if verbose > 0:
        iterable = enumerate(tqdm(contribs, desc='barcodes ids'))
    else:
        iterable = enumerate(contribs)
    for i, contrib in iterable:
        # get lists of round ids from lists of contributing spot
        r_ids = spot_rounds[np.array(contrib)] # need array() for indexing
        # set barcode bits to 1 at round ids of contributing spots
        bcd_sequences[i, r_ids] = 1
        # uncomment to observe what's happening:
        # print(contrib, ' --> ', bcd_sequences[i, :])
        # make network spots-barcodes to record spots contributions
        for spot_id in contrib:
            if spot_id in spots_bcd.keys():
                spots_bcd[spot_id].update({i})  # it' a set, not dictionary
            else:
                spots_bcd[spot_id] = {i}

    return bcd_sequences, spots_bcd


def std_distance(coords):
    """
    Compute the standard deviation of distances of points to their mean center.
    """

    if coords.shape[1] == 2:
        sd = np.sqrt(
            ( np.sum(np.square(coords[:, 0] - coords[:, 0].mean())) \
            + np.sum(np.square(coords[:, 1] - coords[:, 1].mean())) ) / len(coords)
        )
    else:
        sd = np.sqrt(
            ( np.sum(np.square(coords[:, 0] - coords[:, 0].mean())) \
            + np.sum(np.square(coords[:, 1] - coords[:, 1].mean())) \
            + np.sum(np.square(coords[:, 2] - coords[:, 2].mean())) ) / len(coords)
        )
    return sd


def find_barcode_min_dist(barcode, codebook_keys, codebook_vals, max_err=1):
    """
    Find the minimum distance between a barcode and all barcodes in a codebook,
    and infer the identity of this barcode.

    Parameters
    ----------
    barcodes : ndarray
        Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).
    codebook_keys : array
        List of keys of the codebook.
    codebook_vals : ndarray
        2D array of binary values of the codebook.

    Example
    -------
    >>> barcode = np.array([1,1,1,0,0,0,0,0]).reshape((1, -1))
    >>> codebook = {'a': np.array([1,1,1,1,0,0,0,0]),
                    'b': np.array([0,0,1,1,1,1,0,0]),
                    'c': np.array([0,0,0,0,1,1,1,1])}
    >>> cbk_keys, cbk_vals = dict_to_2D_array(codebook)
    >>> find_barcode_min_dist(barcode, cbk_keys, cbk_vals, max_err=1)
    ('a', 1)
    """

    # compute distances between barcode and codebook's barcodes
    dist = cdist(barcode, codebook_vals, metric='cityblock')
    # get the minimun errors / Hamming distance
    min_err = dist.min().astype(int)
    select_codebook = np.argmin(dist)
    # infer identity
    if min_err > max_err:
        bcd_species = None
    else:
        bcd_species = codebook_keys[select_codebook]
    return bcd_species, min_err


def compute_errors(barcodes, codebook_vals):
    """
    Compute the hamming distance between barcode sequences and their closest
    allowed sequence in a codebook.

    Parameters
    ----------
    barcodes : ndarray
        Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).
    codebook_vals : ndarray
        2D array of binary values of the codebook.

    Example
    -------
    >>> barcodes = np.array([[0,1,1,1,1,0,0,0],
                             [1,1,1,0,0,0,0,0],
                             [1,1,0,0,0,1,1,1]])
    >>> codebook = {'a': np.array([1,1,1,1,0,0,0,0]),
                    'b': np.array([0,0,1,1,1,1,0,0]),
                    'c': np.array([0,0,0,0,1,1,1,1])}
    >>> cbk_keys, codebook_vals = dict_to_2D_array(codebook)
    >>> compute_errors(barcodes, codebook_vals)
    (array([2, 1, 3])
    """

    # compute distances between barcode and codebook's barcodes
    errors = cdist(barcodes, codebook_vals, metric='cityblock').astype(int)
    # get the minimun errors / Hamming distance
    min_err = errors.min(axis=1)

    return min_err


def build_barcodes_stats(coords, fit_vars, contribs, barcodes, codebook_keys, 
                         codebook_vals, dist_method="isotropic", max_err=1, verbose=1):
    """
    Make the matrix that holds all informations about all potential barcodes, 
    their contributing spots ids, and derived statistics.

    Parameters
    ----------
    coords : ndarray
        Coordinates of localized spots.
    fit_vars : ndarray
        Results of 3D fits during spots localization. For now the columns are:
        [amplitude,]
    contribs : list(list)
        List of contributing spot ids for each barcode.

    Returns
    -------
    stats : DataFrame
        Statistics of barcodes: mean position [z, y, x], (z dispersion), x/y dispersion,
        mean amplitude, sequences min error to codebook sequences.
    species : list(str)
        Barcodes' species infered from the smallest hamming distance between sequences.
    
    Example
    -------
    >>> coords = np.array([[0, 0, 0],
                           [0, 1, 1],
                           [3, 2, 2],])
    >>> fit_vars = np.array([4, 4, 10]).reshape((-1, 1))
    >>> contribs = [(0, 1), (0, 1, 2)]
    >>> barcodes = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0, 0, 0]])
    >>> codebook = {'a': np.array([0,0,0,0,1,1,1,1]),
                    'b': np.array([1,1,1,1,0,0,0,0]),
                    'c': np.array([1,0,1,0,1,0,1,0])}
    >>> cbk_keys, cbk_vals = dict_to_2D_array(codebook)
    >>> build_barcodes_stats(coords, fit_vars, contribs, barcodes, cbk_keys, cbk_vals)
    (array([[0.        , 0.5       , 0.5       , 1.41421356, 1.15470054,
            4.        , 2.        ],
            [1.        , 1.        , 1.        , 1.41421356, 1.15470054,
            6.        , 1.        ]]),
    ['b', 'b'])
    """

    # make stats
    # for now: centroids mean position, (z dispersion), x/y dispersion, 
    # mean amplitude, std amplitude, sequence errors, species
    if dist_method=="isotropic":
        stats = np.zeros((len(contribs), 7), dtype=float)
    else:
        stats = np.zeros((len(contribs), 8), dtype=float)
    # record species separately to not mess with array of numbers
    species = []
    # we iterate on contribs the same way we did for building barcodes
    if verbose > 0:
        iterable = enumerate(tqdm(contribs, desc='barcodes ids'))
    else:
        iterable = enumerate(contribs)
    for i, contrib in iterable:
        bcd_coords = coords[np.array(contrib), :]
        bcd_fit = fit_vars[np.array(contrib), :]
        mean_pos = bcd_coords.mean(axis=0)
        amp = np.mean(bcd_fit[:, 0])
        amp_std = np.std(bcd_fit[:, 0])
        # TODO: use modified version  to compute that once in the pipeline
        spec, err = find_barcode_min_dist(
            barcodes[i].reshape((1, -1)), 
            codebook_keys, 
            codebook_vals, 
            max_err=max_err,
            )
        if dist_method=="isotropic":
            zxy_disp = std_distance(bcd_coords)
            stats[i, :] = np.array([*mean_pos, zxy_disp, amp, amp_std, err])
        else:
            z_disp = np.std(bcd_coords[:, 0])
            xy_disp = std_distance(bcd_coords[:, 1:3])
            stats[i, :] = np.array([*mean_pos, z_disp, xy_disp, amp, amp_std, err])
        species.append(spec)
    # TODO: use np.unique(array, return_inverse=True) for categorical data
    species = np.array(species)
    if dist_method=="isotropic":
        colnames = ['z', 'y', 'x', 'z/x/y std', 'amplitude', 'amplitude std', 'error']
    else:
        colnames = ['z', 'y', 'x', 'z std', 'x/y std', 'amplitude', 'amplitude std', 'error']
    stats = pd.DataFrame(data=stats, columns=colnames)
    stats['error'] = stats['error'].astype(int)
    stats['species'] = pd.Series(species, dtype='category')

    return stats


def transform_bit_data(data, select=None, r_mean=1, reverse=False):
    """
    Transform amplitudes so that selected spots have a specific mean amplitude, 
    and reverse the min and max of the distribution.
    
    Parameters
    ----------
    r_mean : float
        Target mean of selected spots amplitudes.
    """
    
    data -= data.min()
    # mean of selected spots
    if select is None:
        f_mean = data.mean()
    else:
        f_mean = data[select].mean()
    # max of the whole distribution
    f_max = data.max()
    if reverse:
        # reverse distribution
        data = f_max - data
        # compute the maximum of the reversed distribution
        r_max = r_mean * f_max / (f_max - f_mean)
        # rescale reversed distribution
        data *= r_max / f_max
    else:
        data *= r_mean / f_mean
    
    return data

def recompute_barcodes_amplitudes(fit_vars, contribs, verbose=1):
    """
    Recompute the mean of contributing spots' amplitudes for each barcode

    Parameters
    ----------
    fit_vars : ndarray
        Results of 3D fits during spots localization. For now the columns are:
        [amplitude,]
    contribs : list(list)
        List of contributing spot ids for each barcode.

    Returns
    -------
    amp_mean : array
        Mean amplitudes of barcodes.
    """

    if verbose > 0:
        iterable = enumerate(tqdm(contribs, desc='barcodes ids'))
    else:
        iterable = enumerate(contribs)
    amp_mean = []
    for i, contrib in iterable:
        bcd_fit = fit_vars[np.array(contrib), :]
        new_mean = np.mean(bcd_fit[:, 0])
        # new_std = np.std(bcd_fit[:, 0]) # not used currently
        amp_mean.append(new_mean)
    amp_mean = np.array(amp_mean)

    return amp_mean


def make_empty_stats(dist_params):
    """
    Make an empty stats DataFrame in case no barcode exists.
    """
    
    if isinstance(dist_params, Number):
        colnames = ['z', 'y', 'x', 'z/x/y std', 'amplitude', 'amplitude std', 'error', 'species']
    else:
        colnames = ['z', 'y', 'x', 'z std', 'x/y std', 'amplitude', 'amplitude std', 'error', 'species']
    stats = pd.DataFrame(np.full((0, len(colnames)), None), columns=colnames)
    return stats


def reverse_sigmoid(x, w0, b):
    """ Return the reverse sigmoid of an array"""
    return w0 - x**4 / (b**4 + x**4)

def logistic(x, w0=1, k=1, b=0):
    return w0 / (1 + np.exp(-k * (x - b)))

def compute_barcode_multiplicity(spots_combinations, spots_bcd, 
                                 bcd_per_spot_params, fct=None):
    """
    Compute a loss for each barcode depending on the multiplicity of their
    contributing spots, which is the number of potential barcodes per spot.
    A reverse sigmoid function is applied to the distribution of the maximum
    multiplicities o barcodes to help "ranking" barcodes.
    
    Parameters
    ----------
    bcd_per_spot_params : dict
        Parameters to compute the multiplicity loss. Keys are 'w0' and 'b', and 
        optionally 'weight' to modify its importance relative to other losses.
        The default weight is ~1/3 so this loss has the same importance as spots 
        dispersion and mean amplitude when added to the barcode loss.
    
    Example
    -------
    >>> spots_combinations = [[0, 1, 2, 3], [2, 3, 4, 5]]
    >>> spots_bcd = {0: [0], 1: [0], 2: [0, 1], 3: [0, 1], 4: [1], 5: [1]}
    """
    
    if fct is None:
        fct = np.max
    
    # count the number of barcodes per spot
    spots_multiplicities = {key: len(val) for key, val in spots_bcd.items()}
    # get the max (or min, mean of else) of spots multiplicity per barcode
    bcd_multiplicities = np.array([fct([spots_multiplicities[spot_idx] for spot_idx in combi]) for combi in spots_combinations])
    # compute the reverse sigmoid of the multiplicities
    if 'weight' in bcd_per_spot_params.keys():
        weight = bcd_per_spot_params['weight']
    else:
        weight = 0.333
    loss_multiplicities = logistic(
        bcd_multiplicities, 
        w0=bcd_per_spot_params['w0'], 
        k=1, 
        b=bcd_per_spot_params['b'],
        ) * weight
    return loss_multiplicities


def remap_spots_bcd(spots_bcd, select, size=None, verbose=1):
    """
    Modify the values of a dictionary given a boolean array that filters this
    dictionary.

    Parameters
    ----------
    spots_bcd : dict
        For each spot, all barcodes it contributed to.
    select : array(bool)
        Filters the list of spots combinations making the barcodes.
    size : int, optional, default None
        Provide a known count of selected elements to avoid computing it.

    Example
    -------
    If we start with:
    >>> spots_combinations = [(0, 1, 2), (1, 2), (0, 1, 3), (4, 5), (3, 6)]
    >>> spots_bcd = {0: [0, 2], 1: [0, 1, 2], 2: [0, 1], 3:[2, 4], 4: [3], 5: [3], 6: [4]}
    and we filter `spots_combinations` with a bolean array
    >>> select = np.array([True, False, True, False, True])
    this results in `spots_combinations = [(0, 1, 2), (0, 1, 3), (3, 6)]`
    To remap the values of `spot_bcd`, we use:
    >>> remap_dict(spots_bcd, select)
    {0: [0, 1], 1: [0, 1], 2: [0], 3: [1, 2], 4: [], 5: [], 6: [2]}
    """

    new_spots_bcd = {}
    old_ids = np.where(select)[0]
    if size is None:
        size = select.sum()
    new_ids = np.arange(size)
    mapper = dict(zip(old_ids, new_ids))
    if verbose > 1:
        iterable = tqdm(spots_bcd.keys(), desc='spots ids', leave=False)
    else:
        iterable = spots_bcd.keys()
    for key in iterable:
        # apply the mapping from old to new barcode ids to each dictionary value
        new_spots_bcd[key] = [mapper[i] for i in spots_bcd[key] if i in mapper.keys()]
    
    return new_spots_bcd


# def filter_barcodes(sequences, stats, combinations, spots_bcd=None, 
#                     max_err=1, err_col=-1, verbose=1):
#     """
#     Filter the barcodes statistics array given barcodes sequences errors.
#     """
    
#     select = stats[:, err_col] <= max_err
#     sequences = sequences[select, :]
#     stats = stats[select, :]
#     combinations = list(itertools.compress(combinations, select))
#     # remap spots_bcd
#     if spots_bcd is not None:
#         if verbose > 0:
#             print("Remapping spots --> barcodes dictionary")
#         new_spots_bcd = remap_spots_bcd(
#             spots_bcd, 
#             select, 
#             size=len(stats), 
#             verbose=verbose,
#             )
#         return sequences, stats, combinations, new_spots_bcd, select
    
#     return sequences, stats, combinations, select


# def filter_max_bcd_per_spot(max_bcd_per_spot, bcd_sequences, spots_combinations, spots_bcd):
#     """
#     Filter barcodes and spots when the latter have too many related barcodes.
    
#     Parameters
#     ----------
#     max_bcd_per_spot : int
#         Maximum number of barcodes spots can be related to before being filtered.
#     bcd_sequences : array
#         Barcode sequences from all possible spots ids combinations.
#     spots_combinations : list(list)
#         For each barcode, its contributing spots.
#     spots_bcd : dict
#         For each spot, all barcodes it contributed to.

#     Returns
#     -------
#     bcd_sequences : array
#         Barcode sequences from all possible spots ids combinations.
#     spots_combinations : list(list)
#         For each barcode, its contributing spots.
#     spots_bcd : dict
#         For each spot, all barcodes it contributed to.
#     """
    
#     select_spots = np.full(len(spots_bcd.keys()), True)
#     select_bcds = np.full(len(bcd_sequences), True)
    
#     for spot_id, linked_barcodes in spots_bcd.items():
#         if len(linked_barcodes) > max_bcd_per_spot:
#             select_spots[spot_id] = False
#             select_bcds[tuple(linked_barcodes)] = False
            
#     bcd_sequences = bcd_sequences[select_bcds]
#     spots_combinations = list(itertools.compress(spots_combinations, select))
#     # remap spots_bcd
#     if spots_bcd is not None:
#         if verbose > 0:
#             print("Remapping spots --> barcodes dictionary")
#         new_spots_bcd = remap_spots_bcd(
#             spots_bcd, 
#             select, 
#             size=len(bcd_sequences), 
#             )
#         return bcd_sequences, spots_combinations, new_spots_bcd, select
    
#     return bcd_sequences, spots_combinations, select


def prefilter_barcodes_error(bcd_sequences, codebook_vals, spots_combinations,  
                             spots_bcd=None, max_err=1, verbose=1):
    """
    Filter the arrays of barcodes, contributing spots and updating the 
    spot-barcode network given barcodes sequences errors.
    """
    
    errors = compute_errors(bcd_sequences, codebook_vals)
    select = errors <= max_err
    bcd_sequences = bcd_sequences[select]
    errors = errors[select]
    spots_combinations = list(itertools.compress(spots_combinations, select))
    # remap spots_bcd
    if spots_bcd is not None:
        if verbose > 0:
            print("Remapping spots --> barcodes dictionary")
        new_spots_bcd = remap_spots_bcd(
            spots_bcd, 
            select, 
            size=len(bcd_sequences), 
            verbose=verbose,
            )
        return bcd_sequences, spots_combinations, new_spots_bcd, errors, select
    
    return bcd_sequences, spots_combinations, errors, select


def filter_barcodes_array(data, threshold, bcd_sequences, spots_combinations, 
                          stats, direction='greater', spots_bcd=None,
                          stats_norm=None, verbose=1):
    """
    Filter barcodes and several related variable, and update the 
    spot-barcode network given an array of values and a hard threshold.
    """
    
    if direction == 'greater':
        select = data > threshold
    else:
        select = data < threshold
    bcd_sequences = bcd_sequences[select]
    spots_combinations = list(itertools.compress(spots_combinations, select))
    stats = stats.loc[select, :]
    results = [bcd_sequences, spots_combinations, stats]
    # remap spots_bcd
    if spots_bcd is not None:
        if verbose > 0:
            print("Remapping spots --> barcodes dictionary")
        new_spots_bcd = remap_spots_bcd(
            spots_bcd, 
            select, 
            size=len(bcd_sequences), 
            verbose=verbose,
            )
        results.append(new_spots_bcd)
    if stats_norm is not None:
        stats_norm = stats_norm.loc[select, :]
        results.append(stats_norm)
    results.append(select)

    return results

        
def build_barcodes_network_array(combinations, verbose=1):
    """
    Build the graph of barcodes, where nodes are barcode ids and edges represent 
    common contributing spot ids. 
    This is the version for array-based networks.

    Example
    -------
    >>> combinations = [(0, 1, 2), (1, 2), (0, 1), (4, 5), (3, 6)]
    >>> build_barcodes_network(combinations)
    array([[0, 1],
           [0, 2],
           [1, 2],
           [2, 4]])
    """

    n_bcd = len(combinations)
    pairs = []
    # starting from all barcode ('source')
    if verbose > 0:
        iterable = enumerate(tqdm(combinations, desc='barcodes ids'))
    else:
        iterable = enumerate(combinations)
    for bcd_src_id, combination in iterable:
        # for each contributing spot id
        for spot_id in combination:
            # look in all following barcode targets
            for bcd_trg_id in range(bcd_src_id + 1, n_bcd):
                if spot_id in combinations[bcd_trg_id]:
                    pairs.append([bcd_src_id, bcd_trg_id])
    if len(pairs) == 0:
        pairs.append([])
    pairs = np.vstack(pairs)
    # remove duplicate edges
    if verbose > 1:
        print("    Removing duplicate pairs")
    pairs = np.unique(pairs, axis=0)

    return pairs


def build_barcodes_network(spots_combinations, spots_bcd, verbose=1):
    """
    Build the graph of barcodes, where nodes are barcode ids and edges represent 
    common contributing spot ids.

    Parameters
    ----------
    spots_combinations : list(list)
        For each barcode, its contributing spots.
    spots_bcd : dict
        For each spot, all barcodes it contributed to.

    Example
    -------
    >>> spots_combinations = [(0, 1, 2), (1, 2), (0, 1, 3), (4, 5), (3, 6)]
    >>> spots_bcd = {0: [0, 2], 1: [0, 1, 2], 2: [0, 1], 3:[2, 4], 4: [3], 5: [3], 6: [4]}
    >>> build_barcodes_network(spots_combinations)
    {0: {1, 2}, 1: {0, 2}, 2: {0, 1, 4}, 4: {2}}
    """

    pairs = {}
    # starting from all barcode ('source')
    if verbose > 0:
        iterable = enumerate(tqdm(spots_combinations, desc='barcodes ids'))
    else:
        iterable = enumerate(spots_combinations)
    for bcd_src_id, combination in iterable:
        # for each contributing spot id
        neighbs = []
        for spot_id in combination:
            neighbs.extend(list(spots_bcd[spot_id]))
        neighbs = set(neighbs)
        neighbs.remove(bcd_src_id)
        if len(neighbs) != 0:
            pairs[bcd_src_id] = neighbs

    return pairs


def build_barcodes_network_trim(
    bcd_sequences, 
    spots_combinations, 
    spots_bcd,
    stats,
    stats_norm,
    verbose=1,
    ):
    """
    Look for connected barcodes, and discard the connected ones with the worst loss.
    Update related barcodes objects.

    Parameters
    ----------
    spots_combinations : list(list)
        For each barcode, its contributing spots.
    spots_bcd : dict
        For each spot, all barcodes it contributed to.

    Example
    -------
    >>> bcd_sequences = np.array([[1, 1, 0, 0],
                                  [0, 1, 1, 0],
                                  [1, 1, 1, 0],
                                  [0, 0, 1, 1],
                                  [0, 0, 1, 1]])
    >>> spots_combinations = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6)]
    >>> spots_bcd = {0: [0], 1: [0, 1], 2: [1, 4], 3: [2], 4: [2, 3], 5: [3], 6: [4]}
    >>> stats = pd.DataFrame(np.array([4, 3, 2, 1, 0]), columns=['loss'])
    >>> stats_norm = stats.copy()
    >>> pairs, bcd_sequences, spots_combinations, stats, stats_norm = build_barcodes_network(
            bcd_sequences, spots_combinations, spots_bcd, stats, stats_norm)
    >>> spots_combinations
    [(4, 5), (2, 6)]
    """

    bcd_keep = set()
    discarded = set()
    # starting from all barcode ('source')
    # need to iterate in the order of best barcodes (lowest loss)
    losses = stats_norm.loc[:, 'loss'].values
    ordered_bcds = np.argsort(losses)
    if verbose > 0:
        iterable = tqdm(ordered_bcds, desc='barcodes ids')
    else:
        iterable = ordered_bcds
    for bcd_src_id in iterable:
        combination = spots_combinations[bcd_src_id]
        if bcd_src_id not in discarded:
            # get all connected barcodes to the source barcode
            neighbs = []
            for spot_id in combination:
                neighbs.extend(list(spots_bcd[spot_id]))
            # remove barcodes in `discarded` from the neighbors
            neighbs = set(neighbs).difference(discarded)
            neighbs_arr = np.array(list(neighbs))
            # if len(neighbs_arr) > 0:
            neighbs_losses = losses[neighbs_arr]
            best_bcd = neighbs_arr[np.argmin(neighbs_losses)]
            bcd_keep.add(best_bcd)
            neighbs.remove(best_bcd)
            bcd_keep = bcd_keep.difference(neighbs)
            discarded.update(neighbs)
    # filter corresponding objects
    bcd_keep = np.array(list(bcd_keep))
    select = np.full(len(bcd_sequences), False)
    select[bcd_keep] = True
    bcd_sequences = bcd_sequences[select]
    spots_combinations = list(itertools.compress(spots_combinations, select))
    # no need of spots_bcd anymore 
    stats = stats.loc[select, :]
    stats_norm = stats_norm.loc[select, :]

    # set `pairs` to None to signal to further functions that barcodes have been filtered
    pairs = None

    return pairs, bcd_sequences, spots_combinations, stats, stats_norm


class PercentileRescaler(BaseEstimator, TransformerMixin):
    """
    Rescale values between 0 and 1 for each column using a low and high
    percentile bound for each column.
    If percentile range is given for each colums, it must be as a list or array 
    if data is an array, or as a dictionnary too if data is a dataframe, 
    with keys corresponding to the dataframe's columns.
    
    Example
    -------
    >>> data = np.random.uniform(low=0, high=1, size=(1000, 5))
    >>> data[:, 4] = 0.8
    >>> data *= np.array([[1, 3, 0.5, 6, 1]])
    >>> data += np.array([[0, -1.5, 40, -10, 0]])
    >>> X_train = data[:700, :]
    >>> X_test = data[-300:, :]
    >>> # for i in range(data.shape[-1]):
    >>> #     plt.hist(X_train[:, i], alpha=0.5)
    >>> perc_rescaler = PercentileRescaler()
    >>> y_train = perc_rescaler.fit_transform(X_train)
    >>> y_test = perc_rescaler.transform(X_test)
    >>> fig, axs = plt.subplots(2, 2)
    >>> axs[0, 0].boxplot(X_train)
    >>> axs[0, 0].set_title('X_train')
    >>> axs[0, 1].boxplot(X_test)
    >>> axs[0, 1].set_title('X_test')
    >>> axs[1, 0].boxplot(y_train)
    >>> axs[1, 0].set_title('y_train')
    >>> axs[1, 1].boxplot(y_test)
    >>> axs[1, 1].set_title('y_test')
    
    # with a dataframe, from the previous data:
    >>> columns = ['a', 'b', 'c', 'd', 'e']
    >>> df_train = pd.DataFrame(X_train, columns=columns)
    >>> df_test = pd.DataFrame(X_test, columns=columns)
    >>> perc_rescaler = PercentileRescaler()
    # or with specific prcentiles, here as lists:
    >>> rescaler_kwargs = {'perc_low': [0, 0, 0, 0, 0],
    >>>                    'perc_up': [100, 100, 100, 100, 100]}
    >>> perc_rescaler = PercentileRescaler(**rescaler_kwargs)
    >>> y_train = perc_rescaler.fit_transform(df_train)
    >>> y_test = perc_rescaler.transform(df_test)
    >>> fig, axs = plt.subplots(2, 2)
    >>> axs[0, 0].boxplot(X_train)
    >>> axs[0, 0].set_title('df_train')
    >>> axs[0, 1].boxplot(X_test)
    >>> axs[0, 1].set_title('df_test')
    >>> axs[1, 0].boxplot(y_train)
    >>> axs[1, 0].set_title('y_train')
    >>> axs[1, 1].boxplot(y_test)
    >>> axs[1, 1].set_title('y_test')
    """
    
    def __init__(self, perc_low=0, perc_up=100):
        self.perc_low = perc_low
        self.perc_up = perc_up
        
    def find_boundaries(self, X, low, up, col_name=None, y=None):
        X = np.copy(X)
        thresh_low = np.percentile(X, low)
        thresh_up = np.percentile(X, up) - thresh_low
        if self.data_type == 'ndarray':
            self.lower_bound.append(thresh_low)
            self.upper_bound.append(thresh_up)  
        elif self.data_type == 'dataframe':
            self.lower_bound[col_name] = thresh_low
            self.upper_bound[col_name] = thresh_up  

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            self.data_type = 'ndarray'
            self.lower_bound = []
            self.upper_bound = []
            if isinstance(self.perc_low, Number):
                self.perc_low = [self.perc_low] * X.shape[1]
                self.perc_up = [self.perc_up] * X.shape[1]
            else:
                assert(X.shape[1] == len(self.perc_low))
                assert(X.shape[1] == len(self.perc_up))
            for i in range(X.shape[1]):
                if len(np.unique(X[:, i])) == 1:
                    # if variable has a unique value
                    self.lower_bound.append(0.5)
                    self.upper_bound.append(0.5)     
                else:
                    self.find_boundaries(X[:, i], self.perc_low[i], self.perc_up[i])
        elif isinstance(X, pd.DataFrame):
            self.data_type = 'dataframe'
            self.lower_bound = {}
            self.upper_bound = {}
            self.columns = X.columns
            if isinstance(self.perc_low, Number):
                self.perc_low = {col: self.perc_low for col in self.columns}
                self.perc_up = {col: self.perc_up for col in self.columns}
            elif isinstance(self.perc_low, (list, np.ndarray)):
                self.perc_low = {col: self.perc_low[i] for i, col in enumerate(self.columns)}
                self.perc_up = {col: self.perc_up[i] for i, col in enumerate(self.columns)}
            elif isinstance(self.perc_low, dict):
                assert(X.shape[1] == len(self.perc_low.keys()))
                assert(X.shape[1] == len(self.perc_up.keys()))
            for i in self.columns:
                if len(np.unique(X.loc[:, i])) == 1:
                    # if variable has a unique value
                    self.lower_bound[i] = 0.5
                    self.upper_bound[i] = 0.5
                else:
                    self.find_boundaries(X.loc[:, i].values, self.perc_low[i], self.perc_up[i], col_name=i)
        return self
    
    def transform(self, X, y=None):
        if self.data_type == 'ndarray':
            X = np.copy(X)
            for i in range(X.shape[1]):
                x = np.copy(X[:, i])
                if self.lower_bound[i] == self.upper_bound[i]:
                    # this variable had a unique value
                    x[:] = self.lower_bound[i]
                else:
                    x[x < self.lower_bound[i]] = self.lower_bound[i]
                    x = x - self.lower_bound[i]
                    x[x > self.upper_bound[i]] = self.upper_bound[i]
                    if self.upper_bound[i] > 0:
                        x = x / self.upper_bound[i]
                X[:, i] = x
        elif self.data_type == 'dataframe':
            X = X.copy()
            for i in self.columns:
                x = np.copy(X.loc[:, i].values)
                if self.lower_bound[i] == self.upper_bound[i]:
                    # this variable had a unique value
                    x[:] = self.lower_bound[i]
                else:
                    x[x < self.lower_bound[i]] = self.lower_bound[i]
                    x = x - self.lower_bound[i]
                    x[x > self.upper_bound[i]] = self.upper_bound[i]
                    if self.upper_bound[i] > 0:
                        x = x / self.upper_bound[i]
                X.loc[:, i] = x
        return X
    

def normalize_stats(stats, rescaler=None, rescaler_kwargs=None, reverse_cols=['amplitude']):
    """
    Perform data transformation on barcode statistics to make variable more
    comparable to each other, so we can use more meaningfull weights.

    Parameters
    ----------
    stat : DataFrame
        (z dispersion), x/y dispersion, mean amplitude, std amplitude, sequence error.
        There is no mean_z, mean_y or mean_x, but if that's the case they are trimmed.
    """
    
    if 'z' in stats.columns:
        # discard coordinates of barcodes
        stats_norm = stats.drop(columns=['z', 'y', 'x']).copy()
    else:
        stats_norm = stats.copy()

    num_cols = [x for x in stats_norm if x != 'species']
    if rescaler is None:
        if rescaler_kwargs is None:
            if 'z std' in stats_norm.columns:
                rescaler_kwargs = {'perc_low': [0, 0, 0, 0, 0],
                                   'perc_up': [100, 100, 100, 100, 100]}
            else:
                rescaler_kwargs = {'perc_low': [0, 0, 0, 0],
                                   'perc_up': [100, 100, 100, 100]}
        rescaler = PercentileRescaler(**rescaler_kwargs)
        stats_norm.loc[:, num_cols] = rescaler.fit_transform(stats_norm.loc[:, num_cols])
    else:
        stats_norm.loc[:, num_cols] = rescaler.transform(stats_norm.loc[:, num_cols])
    
    # reverse variables 0 <--> 1
    # here mean amplitude only
    if reverse_cols is not None:
        stats_norm.loc[:, reverse_cols] = 1 - stats_norm.loc[:, reverse_cols]
        
    return stats_norm, rescaler


def compute_individual_losses(stats, weights, inplace=False):
    """
    Compute the contribution to loss of each individual barcode.

    Parameters
    ----------
    stat : DataFrame
        (z dispersion), x/y dispersion, mean amplitude, std amplitude, sequence error, species.
        There is no mean_z, mean_y or mean_x.
    weights : array
        Coefficients of each variable in stats for the loss. The last coefficient
        is for the number of barcodes in the combination considered.
    inplace : bool
        If True, stack the loss to the stats array and return stats.
    """

    num_cols = [x for x in stats if x != 'species']
    loss = stats.loc[:, num_cols].values @ weights[:-1].reshape((-1, 1))
    if inplace:
        stats['loss'] = loss
        return stats
    return loss


def compute_selection_loss(indiv_loss, weights, loss_params, fct_aggreg=np.mean):
    """
    Parameters
    ----------
    indiv_loss : darray
        Individual losses of barcodes.
    weights : array
        Coefficients of each variable in stats for the loss. The last coeafficient
        is for the number of barcodes in the combination considered.
    loss_params : dict
        General informations and parameters to parametrize the loss.
    """
    
    # Contribution of selection size to the loss
    loss_size = np.abs( (loss_params['n_bcd_mean'] - len(indiv_loss)) / loss_params['n_bcd_mean'] )
    loss = fct_aggreg(indiv_loss) + loss_size * weights[-1]
    return loss


def powerset(iterable, size_min=1, size_max=None):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    if size_max is None:
        size_max = len(iterable)
    return itertools.chain.from_iterable(itertools.combinations(iterable, r) for r in range(size_min, size_max+1))


def is_valid_array(bcd_combination, pairs, n_bcd_min, n_bcd_max):
    """
    Check if a combination of selected barcodes is valid, i.e. barcode ids are not
    connected in the barcode network due to common contributing spots, and the
    number of selected barcode is within a size range.
    This is the version for array-based networks.

    Example
    -------
    >>> bcd_combination = (0, 1, 2)
    >>> pairs = np.array([[0, 3],
                          [1, 4]])
    >>> is_valid(bcd_combination, pairs)
    True
    >>> bcd_combination = (0, 1, 2)
    >>> pairs = np.array([[0, 3],
                          [1, 2]])
    >>> is_valid(bcd_combination, pairs)
    False
    """

    if n_bcd_max < len(bcd_combination) < n_bcd_min:
        return False
    for i, bcd_src in enumerate(bcd_combination):
        select_src = np.logical_or(pairs[:, 0] == bcd_src, pairs[:, 1] == bcd_src)
        for bcd_trg in bcd_combination[i+1:]:
            select_trg = np.logical_or(pairs[:, 0] == bcd_trg, pairs[:, 1] == bcd_trg)
            select_common = np.logical_and(select_src, select_trg)
            if np.any(select_common):
                return False
    return True


def is_valid(bcd_combination, pairs, n_bcd_min=1, n_bcd_max=np.inf):
    """
    Check if a combination of selected barcodes is valid, i.e. barcode ids are not
    connected in the barcode network due to common contributing spots, and the
    number of selected barcode is within a size range.

    Example
    -------
    >>> bcd_combination = (0, 1, 2)
    >>> pairs = {0: {3}, 3: {0}, 1: {4}, 4: {1}}
    >>> is_valid(bcd_combination, pairs)
    True
    >>> bcd_combination = (0, 1, 2)
    >>> pairs = {0: {3}, 3: {0}, 1: {2}, 2: {1}}
    >>> is_valid(bcd_combination, pairs)
    False
    """

    if (len(bcd_combination) < n_bcd_min) or (len(bcd_combination) > n_bcd_max):
        return False
    for i, bcd_src in enumerate(bcd_combination):
        if bcd_src in pairs.keys():
            for bcd_trg in pairs[bcd_src]:
                if bcd_trg in bcd_combination:
                    return False
    return True


def find_best_barcodes_combination(stats, pairs, loss_function, weights, 
                                   loss_params=None, n_best=1, n_bcd_min=1, 
                                   n_bcd_max=None, n_bcd_combis=None, verbose=1):
    """
    Try all combinations of barcodes and save the one with the best loss function.
    """

    n_bcd = len(stats)
    # best_losses = np.full(shape=n_best, fill_value=np.inf) # TODO: save n_best combis
    best_loss = np.inf
    best_combi = np.full(n_bcd, False)
    bcd_ids = np.arange(n_bcd)
    if verbose > 2:
        print("    Generating barcodes combinations")
    bcd_combinations = powerset(bcd_ids, n_bcd_min, n_bcd_max)
    if verbose > 0 and n_bcd_combis is not None:
        iterable = enumerate(tqdm(bcd_combinations, total=n_bcd_combis, desc='Computing best barcodes combination'))
    else:
        iterable = enumerate(bcd_combinations)

    for i, bcd_combination in iterable:
        if is_valid(bcd_combination, pairs, n_bcd_min, n_bcd_max):
            # compute loss
            loss = loss_function(stats.loc[bcd_combination, 'loss'], weights, loss_params)
            if loss < best_loss:
                best_loss = loss
                best_combi = bcd_combination
    return best_combi, best_loss


def clean_selection_array(bcd_select, pairs, new_candidates=None):
    """
    Correct a selection of barcodes by eliminating barcodes linked to each other
    in their network (`pairs`), while potentially iterating in a specific order.
    This is the version for array-based networks.

    Parameters
    ----------
    bcd_select : array(bool)
        Selected barcodes.
    pairs : ndarray
        Matrix linking pairs of barcodes by their id.
    new_candidates : array(int) or None (default)
        If not None, specific order to iterate over barcodes and eliminate their 
        neighbors (given by `pairs`) in the selection.
    
    Returns
    -------
    bcd_select : array(bool)
        Cleaned barcodes selection.
    
    Example
    -------
    >>> bcd_select = np.array([True, True, False, False, True, True, True]) 
    >>> pairs = np.array([[0, 2],
                          [1, 4],
                          [2, 5],
                          [5, 6]])
    >>> new_candidates = np.array([4, 5])
    >>> clean_selection(bcd_select, pairs, new_candidates)
    array([ True, False, False, False,  True,  True, False])
    """

    if new_candidates is None:
        new_candidates = np.arange(len(bcd_select))
    for i in new_candidates:
        # avoid considering new candidates filtered out by previous new candidates
        if bcd_select[i]:
            neighb_ids = np.hstack((pairs[pairs[:, 0] == i, 1],
                                    pairs[pairs[:, 1] == i, 0]))
            if len(neighb_ids) > 0:
                bcd_select[neighb_ids] = False
    return bcd_select


def clean_selection(bcd_select, pairs, new_candidates=None):
    """
    Correct a selection of barcodes by eliminating barcodes linked to each other
    in their network (`pairs`), while potentially iterating in a specific order.

    Parameters
    ----------
    bcd_select : array(bool)
        Selected barcodes.
    pairs : ndarray
        Matrix linking pairs of barcodes by their id.
    new_candidates : array(int) or None (default)
        If not None, specific order to iterate over barcodes and eliminate their 
        neighbors (given by `pairs`) in the selection.
    
    Returns
    -------
    bcd_select : array(bool)
        Cleaned barcodes selection.
    
    Example
    -------
    >>> bcd_select = np.array([True, True, False, False, True, True, True]) 
    >>> pairs = {0: {2}, 2: {0, 5}, 1: {4}, 4: {1}, 5: {2, 6}, 6: {5}}
    >>> new_candidates = np.array([4, 5])
    >>> clean_selection(bcd_select, pairs, new_candidates)
    array([ True, False, False, False,  True,  True, False])
    """

    if new_candidates is None:
        new_candidates = np.arange(len(bcd_select))
    for i in new_candidates:
        # avoid considering new candidates filtered out by previous new candidates
        if bcd_select[i] and i in pairs.keys():
            neighb_ids = pairs[i]
            if len(neighb_ids) > 0:
                bcd_select[list(neighb_ids)] = False
    return bcd_select


def search_stochastic_combinations(stats, pairs, loss_function, weights, 
                                   loss_params=None, maxiter=200, patience='maxiter',
                                   min_candidates=None, max_candidates=None, mean_candidates=None,
                                   history=False, initialize='maxloss', 
                                   propose_method='single_step', n_repeats=1, verbose=1):
    """
    Search the best set of barcodes with random combinations.
    """

    rng = default_rng()
    # in the future: subdivide optimization in as many local networks as possible
    n_bcd = len(stats)
    if min_candidates is None:
        min_candidates = 1
    if max_candidates is None:
        max_candidates = n_bcd
    if propose_method == 'iter_bcd':
        maxiter = n_bcd * n_repeats
    elif propose_method == 'single_step':
        maxiter = 0
    if patience == 'maxiter':
        patience = maxiter
    batch_size = int(maxiter / n_repeats)
    # best_losses = np.full(shape=n_best, fill_value=np.inf) # TODO: save n_best combis
    bcd_ids = np.arange(len(stats))
    if history:
        record_new_candidates = []
        record_selection_test = []
        record_selection = []
        record_loss= []
        record_decision = []
    if verbose > 2:
        print("    Searching best barcodes combination")
    
    # initialize barcodes candidates
    bcd_select = np.full(n_bcd, False)
    # First get the number of proposed candidates...
    if 0 < max_candidates < 1:
        # use proportion of barcodes instead of absolute number
        max_propose = int(max_candidates * n_bcd)
    else:
        max_propose = max_candidates
    n_propose = rng.integers(min_candidates, max_propose+1)
    # ... then select this number of candidates
    if initialize == 'maxloss':
        # get sorted array of best barcodes from individual losses
        new_candidates = np.argsort(stats['loss']) # increasing order of individual loss
    elif initialize == 'minloss':
        # get only `min_candidates` barcodes
        new_candidates = np.argsort(stats['loss'])[:min_candidates]
    elif initialize == 'meanloss':
        # get only `mean_candidates` barcodes
        new_candidates = np.argsort(stats['loss'])[:int(mean_candidates)]
    elif initialize == 'maxrand':
        # random initialization with all barcodes
        new_candidates = rng.choice(bcd_ids, size=n_bcd, replace=False)
    elif initialize == 'minrand':
        # random initialization with `n_propose` barcodes
        new_candidates = rng.choice(bcd_ids, size=n_propose, replace=False)
    else:
        raise ValueError(f'the argument initialize={initialize} is not valid')
    bcd_select[new_candidates] = True
    if verbose > 1:
        print(f'proposing {bcd_select.sum()} candidates')
        print('total number of barcodes:', n_bcd)
    # clean selection using pairs iterating on array of best barcodes
    if pairs is not None:
        # pairs is None only if propose_method == 'single_step', so maxiter == 0
        bcd_select = clean_selection(bcd_select, pairs, new_candidates)
    if verbose > 1:
        print(f'selected {bcd_select.sum()} candidates initially')
    # compute loss, if the best save selection
    loss = loss_function(stats.loc[bcd_select, 'loss'], weights, loss_params)
    if history:
        record_new_candidates.append(new_candidates.copy())
        record_selection_test.append(bcd_select.copy())
        record_selection.append(bcd_select.copy())
        record_loss.append(loss)
        record_decision.append(True)
    best_loss = loss
    best_combi = bcd_select.copy()

    # Stochastic search
    patience_counter = 0
    if verbose > 1:
        iterable = tqdm(range(maxiter), desc='search round')
    else:
        iterable = range(maxiter)
    for k in iterable:
        i = k % batch_size
        if propose_method == 'iter_bcd':
            # pick the i_th barcode, reverse its selection status, clean the selection
            # if selection is now ON, compute loss, if it's better confirm this action.
            bcd_select_test = bcd_select.copy()
            # reverse selection status
            bcd_select_test[i] = ~bcd_select_test[i]
            if bcd_select_test[i]:
                bcd_select_test = clean_selection(bcd_select_test.copy(), pairs, [i])
            # compute loss, if it improves approve selection
            loss_test = loss_function(stats.loc[bcd_select_test, 'loss'], weights, loss_params)
            proposal_accepted = loss_test < loss
            if proposal_accepted:
                bcd_select[i] = ~bcd_select[i]   # avoid copying bcd_select_test
                loss = loss_test
                if verbose > 2:
                    print(f'barcode {i} switched to {bcd_select[i]}')
            # for history:
            new_candidates = [i]
        elif propose_method == 'from_all':
            # pick n_propose barcodes among all barcodes
            bcd_select = np.full(n_bcd, False)
            if max_candidates < 1:
                max_propose = int(max_candidates * n_bcd)
            else:
                max_propose = max_candidates
            n_propose = rng.integers(min_candidates, max_propose+1)
            new_candidates = rng.choice(bcd_ids, size=n_propose, replace=False)
            bcd_select[new_candidates] = True
            # clean selection iterating over new barcodes
            bcd_select = clean_selection(bcd_select.copy(), pairs, new_candidates)
            # compute loss, if the best save selection
            loss = loss_function(stats.loc[bcd_select, 'loss'], weights, loss_params)
            # for history latter on
            bcd_select_test = []
            proposal_accepted = True
        elif propose_method == 'from_off':
            # pick n_propose barcodes among shuffled non selected barcodes
            potential_candidates = np.where(~bcd_select)[0]
            n_pot = len(potential_candidates)
            if max_candidates < 1:
                n_propose = min(n_pot, int(max_candidates * n_pot))
            else:
                n_propose = min(n_pot, max_candidates)
            if n_propose == 0:
                results = {
                    'best_combi': best_combi,
                    'best_loss': best_loss,
                }
                if history:
                    results['record_new_candidates'] = record_new_candidates
                    results['record_selection_test'] = record_selection_test
                    results['record_selection'] = record_selection
                    results['record_loss'] = record_loss
                    results['record_decision'] = record_decision
                    results['all_stats'] = stats
                return results
            new_candidates = rng.choice(potential_candidates, n_propose, replace=False)
            # update barcodes selection
            bcd_select[new_candidates] = True
            # clean selection iterating over new barcodes
            bcd_select = clean_selection(bcd_select.copy(), pairs, new_candidates)
            # compute loss, if the best save selection
            loss = loss_function(stats.loc[bcd_select, 'loss'], weights, loss_params)
            # for history latter on
            bcd_select_test = []
            proposal_accepted = True
        if history:
            record_new_candidates.append(new_candidates)
            record_selection_test.append(np.copy(bcd_select_test))
            record_selection.append(np.copy(bcd_select))
            record_loss.append(loss)
            record_decision.append(proposal_accepted)
        if loss < best_loss:
            best_loss = loss
            best_combi = bcd_select
            patience_counter = 0
        else:
            # if loss hasn't improved for n=`patience` iterations, end optimization
            patience_counter += 1
            if patience_counter > patience:
                if verbose > 1:
                    print("Finish optimization because patience exhausted")
                results = {
                    'best_combi': best_combi,
                    'best_loss': best_loss,
                }
                if history:
                    results['record_new_candidates'] = record_new_candidates
                    results['record_selection_test'] = record_selection_test
                    results['record_selection'] = record_selection
                    results['record_loss'] = record_loss
                    results['record_decision'] = record_decision
                    results['all_stats'] = stats
                return results
            
    if verbose > 1:
        print(f"Finish optimization because maxiter {maxiter} reached")
    results = {
        'best_combi': best_combi,
        'best_loss': best_loss,
    }
    if history:
        results['record_new_candidates'] = record_new_candidates
        results['record_selection_test'] = record_selection_test
        results['record_selection'] = record_selection
        results['record_loss'] = record_loss
        results['record_decision'] = record_decision
        results['all_stats'] = stats
    return results
    

def optimize_spots(coords, 
                   fit_vars, 
                   spot_ids, 
                   spot_rounds, 
                   dist_params, 
                   codebook,
                   n_pos_bits=None, 
                   n_bits=None, 
                   size_bcd_min=None, 
                   size_bcd_max=None, 
                   err_corr_dist=1, 
                   weights='auto', 
                   max_positive_bits=0.5, 
                   barcodes_exploration='stochastic', 
                   n_bcd_min_coef=0.75, 
                   n_bcd_max_coef=2, 
                   maxiter=200, 
                   patience='maxiter', 
                   max_candidates='auto', 
                   n_repeats=1,
                   filter_intensity=None, 
                   filter_loss=None, 
                   max_bcd_per_spot=None, 
                   bcd_per_spot_params=None, 
                   rescale_used_spots=True, 
                   n_steps_rescale_used=10, 
                   rescaler=None, 
                   rescaler_kwargs=None,
                   propose_method='single_step', 
                   initialize='maxloss', 
                   trim_network=True, 
                   history=False, 
                   return_extra=False, 
                   return_contribs=False, 
                   return_barcodes_loss=False, 
                   verbose=1):
    """
    
    Parameters
    ----------
    max_positive_bits : float, int
        Ratio or number of allowed bits wehere neighbors are found for a given spot
        before being filtered.
    rescale_used_spots : bool
        If True, spots amplitude are iteratively rescaled given the amplitude of 
        spots used to build barcodes.
    n_steps_rescale_used : int
        Number of iterations to rescale spots amplitudes.
    """

    # check parameters
    if fit_vars.ndim == 1:
        if verbose > 1:
            print("Reshaping fit_vars")
        fit_vars = fit_vars.reshape((-1, 1))
    assert(fit_vars.shape[1] == 1)  # currently use only amplitudes
    if isinstance(dist_params, Number):
        dist_method = 'isotropic'
    else:
        dist_method = 'xy_z_orthog'
    if dist_method=="isotropic":
        assert(isinstance(weights, str) or len(weights) == 5)
    else:
        assert(isinstance(weights, str) or len(weights) == 6)
    int_types = [int, np.intc, np.uintc, np.int_, np.uint, np.longlong, np.ulonglong]
    if not isinstance(spot_ids.dtype, np.integer):
        spot_ids = spot_ids.astype(np.uintc)
    if not spot_rounds.dtype in int_types:
        spot_rounds = spot_rounds.astype(np.ubyte)
    if rescale_used_spots and trim_network:
        if verbose > 0:
            print("Setting trim_network to False because rescale_used_spots is True")
        trim_network = False
    if not rescale_used_spots:
        # will perform barcode selection once
        n_steps_rescale_used = 1

    if n_pos_bits is None:
        # assume identical number of positive bits across sequences
        # otherwise use min and max like:
        # n_pos_bits_min = min([sum([int(i) for i in seq]) for seq in codebook.values()])
        n_pos_bits = sum([int(i) for i in list(codebook.values())[0]])
    if n_bits is None:
        n_bits = len(list(codebook.values())[0])
    if size_bcd_min is None:
        size_bcd_min = n_pos_bits - err_corr_dist
    if size_bcd_max is None:
        size_bcd_max = n_pos_bits + err_corr_dist
    pos_bit_ratio = n_pos_bits / n_bits
    # define the minimum and maximim sizes of a valid selection of barcodes
    counts = np.unique(spot_rounds, return_counts=True)[1]
    n_bcd_min = int(counts.min() / pos_bit_ratio * n_bcd_min_coef)
    n_bcd_max = int(counts.max() / pos_bit_ratio * n_bcd_max_coef)
    n_bcd_mean = counts.mean() / pos_bit_ratio    # expected number of barcodes
    if max_positive_bits <= 1:
        max_positive_bits = int(max_positive_bits * n_bits)
    if max_candidates == 'auto':
        max_candidates = n_bcd_max
    if verbose > 1:
        print("Optimize decoding with")
        print(f"n_bits: {n_bits}, n_pos_bits: {n_pos_bits}, pos_bit_ratio: {pos_bit_ratio}")
        print(f"size_bcd_min: {size_bcd_min}, size_bcd_max: {size_bcd_max}")
        print(f"n_bcd_min: {n_bcd_min}, n_bcd_max: {n_bcd_max}, n_bcd_mean: {n_bcd_mean}")
        print(f"max_candidates: {max_candidates}")
    
    # detect and store neighbors
    if verbose > 0:
        print("Making neighbors")
    neighbors = make_neighbors_lists(
        coords, 
        spot_ids, 
        spot_rounds, 
        dist_params, 
        max_positive_bits,
        dist_method, 
        verbose=verbose,
        )
    
    # make contributing spots combinations for all potential barcodes
    # filter duplicates (within functions)
    if verbose > 0:
        print("Making spots combinations")
    if verbose > 2:
        print('neighbors size:', len(neighbors))
    spots_combinations = make_combinations(
        neighbors, 
        size_bcd_min=size_bcd_min, 
        size_bcd_max=size_bcd_max, 
        verbose=verbose,
        )

    # build barcode sequences from contributing spots ids
    if verbose > 0:
        print("Building barcode sequences")
    bcd_sequences, spots_bcd = make_barcode_from_contributors(
        spots_combinations, 
        spot_rounds, 
        n_bits=n_bits, 
        verbose=verbose,
        )
    n_bcd = len(bcd_sequences)
    
    # # filter barcodes and spots when the latter have too many barcodes
    # should we do it before or after prefilter_barcodes_error?
    # if max_bcd_per_spot is not None and n_bcd > 0:
    #     bcd_sequences, spots_combinations, stats, spots_bcd, _ = filter_max_bcd_per_spot(
    #         max_bcd_per_spot,
    #         bcd_sequences=bcd_sequences, 
    #         spots_combinations=spots_combinations, 
    #         spots_bcd=spots_bcd,
    #         )
    #     n_bcd = len(bcd_sequences)
    
    # compute barcodes errors and filter out those with too many errors (usually 2)
    if verbose > 0:
        print(f"Filtering barcodes with > {err_corr_dist} errors")
    codebook_keys, codebook_vals = dict_to_2D_array(codebook)
    codebook_vals = array_str_to_int(codebook_vals)

    # print('before', bcd_sequences)
    # print('spots_bcd', spots_bcd)
    # print('spots_combinations', spots_combinations)
    # should we eliminate key with empty values?
    if n_bcd > 0:
        bcd_sequences, spots_combinations, spots_bcd, _, _ = prefilter_barcodes_error(
            bcd_sequences, 
            codebook_vals,
            spots_combinations, 
            spots_bcd,
            max_err=err_corr_dist, 
            verbose=verbose,
            )
        n_bcd = len(bcd_sequences)
    # print('spots_bcd', spots_bcd)
    # print('spots_combinations', spots_combinations)
    # print('after', bcd_sequences)


    # assemble table, add statistics, including barcodes min error
    # TODO: accelerate with CUDA
    if verbose > 0:
        print("Making barcodes statistics")
    if n_bcd > 0:
        stats = build_barcodes_stats(
            coords, 
            fit_vars, 
            spots_combinations, 
            bcd_sequences, 
            codebook_keys, 
            codebook_vals, 
            max_err=err_corr_dist, 
            verbose=verbose,
            )
    
    # Filter barcodes using a hard threshold on mean intensity if provided
    if filter_intensity and n_bcd > 0:
        bcd_sequences, spots_combinations, stats, spots_bcd, _ = filter_barcodes_array(
            data=stats['amplitude'],
            threshold=filter_intensity, 
            direction='greater',
            bcd_sequences=bcd_sequences, 
            spots_combinations=spots_combinations, 
            spots_bcd=spots_bcd,
            stats=stats,
            verbose=verbose,
            )
        n_bcd = len(bcd_sequences)

    # Normalize barcodes statistics to make them comparable for optimization
    if verbose > 0:
        print("Normalizing barcodes statistics")
    if n_bcd > 0:
        stats_norm, rescaler = normalize_stats(
            stats, 
            rescaler=rescaler, 
            rescaler_kwargs=rescaler_kwargs,
            )
    
    # Compute multiplicity losses
    if bcd_per_spot_params is not None:
        if verbose > 0:
            print("Computing individual barcodes multiplicity loss")
        loss_multiplicity = compute_barcode_multiplicity(
            spots_combinations, 
            spots_bcd, 
            bcd_per_spot_params,
            )

    # Compute in advance individual barcodes loss
    if verbose > 0:
        print("Computing individual barcodes loss")
    if weights == 'auto':
        # (z dispersion), x/y dispersion, mean amplitude, std amplitude, sequence error, selection size
        if dist_method == 'isotropic':
            weights = np.array([2, 1, 0, 1, 1])
        else:
            weights = np.array([2, 2, 1, 0, 1, 1])
    if n_bcd > 0:
        stats_norm = compute_individual_losses(stats_norm, weights, inplace=True)
        if bcd_per_spot_params is not None:
            stats_norm.loc[:, 'loss'] = stats_norm.loc[:, 'loss'] + loss_multiplicity

    # Filter barcodes using a hard threshold on barcodes' loss if provided
    # do not use this feature if amplitude is iteratively rescaled on selected spots
    if filter_loss and n_bcd > 0:
        bcd_sequences, spots_combinations, stats, spots_bcd, stats_norm, _ = filter_barcodes_array(
            data=stats_norm['loss'],
            threshold=filter_loss, 
            direction='smaller',
            bcd_sequences=bcd_sequences, 
            spots_combinations=spots_combinations, 
            spots_bcd=spots_bcd,
            stats=stats,
            stats_norm=stats_norm,
            verbose=verbose,
            )
        n_bcd = len(bcd_sequences)

    if len(bcd_sequences) == 0:
        if verbose > 0:
            print("No valid barcode found")
        results = {
            'bcd_sequences': None, # np.zeros((1, n_bits)),
            'stats': make_empty_stats(dist_params), # np.full(np.nan, (1, n_bits)),
            'rescaler': None,
            'n_barcodes': 0,
        }
        if history:
            results['record_new_candidates'] = None
            results['record_selection_test'] = None
            results['record_selection'] = None # np.zeros((1, 1))
            results['record_loss'] = None # np.array([None]).reshape((-1, 1))
            results['record_decision'] = None
            results['all_stats'] = None
        if return_contribs:
            results['spots_combinations'] = []
            results['best_combi'] = np.array([])
        if return_barcodes_loss:
            results['barcodes_loss'] = None
        if return_extra:
            results['pairs'] = None
            results['stats_norm'] = None
            results['spots_bcd'] = None
            results['counts'] = None
            results['n_bcd_min'] = None
            results['n_bcd_max'] = None
            results['n_bcd_mean'] = None
            results['max_candidates'] = None
        return results

    # build network of barcodes from contributing spots, exclusive per round
    if propose_method == 'single_step' and trim_network:
        if verbose > 0:
            print("Trimming network of barcodes")
        # when looking for connected barcodes, discard on the fly barcodes
        # with the worst losses, and update related objects
        pairs, bcd_sequences, spots_combinations, stats, stats_norm = build_barcodes_network_trim(
            bcd_sequences=bcd_sequences, 
            spots_combinations=spots_combinations, 
            spots_bcd=spots_bcd,
            stats=stats,
            stats_norm=stats_norm,
            verbose=verbose,
            )
    else:
        if verbose > 0:
            print("Building network of barcodes")
        pairs = build_barcodes_network(spots_combinations, spots_bcd, verbose=verbose)

    # if no barcode is connected, no need for optimization
    if pairs is not None and len(pairs) == 0:
        best_loss = 0
        if verbose > 0:
            print("Valid barcodes don't need optimization")
        results = {
            'bcd_sequences': bcd_sequences,
            'stats': stats, 
            'rescaler': rescaler,
            'n_barcodes': len(bcd_sequences),
        }
        if history:
            results['record_new_candidates'] = None
            results['record_selection_test'] = None
            results['record_selection'] = None # np.zeros((1, 1))
            results['record_loss'] = None # np.array([None]).reshape((-1, 1))
            results['record_decision'] = None
            results['all_stats'] = stats
        if return_contribs:
            results['spots_combinations'] = spots_combinations
            results['best_combi'] = np.full(len(stats), True)
        if return_barcodes_loss:
            results['barcodes_loss'] = stats_norm['loss']
        if return_extra:
            results['pairs'] = pairs
            results['stats_norm'] = stats_norm
            results['spots_bcd'] = spots_bcd
            results['counts'] = None
            results['n_bcd_min'] = None
            results['n_bcd_max'] = None
            results['n_bcd_mean'] = None
            results['max_candidates'] = None
        return results


    # Optimization: search of barcodes combination that minimizes the loss
    if barcodes_exploration == 'all':
        if verbose > 0:
            print("Optimizing barcodes combinations over all possibilities")
        
        # Compute total number of barcodes combinations
        n_bcd_combis = 0
        for n in range(n_bcd_min, n_bcd_max):
            n_bcd_combis += binom(n_bcd, n)
        if verbose > 1:
            print(f'Among the {n_bcd} barcodes, there are {n_bcd_combis} combinations from sizes {n_bcd_min} to {n_bcd_max}')

        best_combi, best_loss = find_best_barcodes_combination(
            stats_norm, 
            pairs, 
            loss_function=compute_selection_loss, 
            weights=weights, 
            loss_params={'n_bcd_mean': n_bcd_mean},
            n_best=1, 
            n_bcd_min=n_bcd_min,
            n_bcd_max=n_bcd_max,
            n_bcd_combis=n_bcd_combis,
            verbose=verbose,
            )
        
    elif barcodes_exploration == 'stochastic':
        if verbose > 0:
            print("Optimizing barcodes combinations with stochastic search")

        if history:
            record_spots_amps = {}
            record_bcd_amps = {}
            record_bcd_selection = {}
            # to record variables at step 0
            new_fit_vars = fit_vars
            new_amp_mean = stats_norm.loc[:, 'amplitude'].values
            
        for i in range(n_steps_rescale_used):
                
            ssc_results = search_stochastic_combinations(
                stats_norm,
                pairs,
                loss_function=compute_selection_loss, 
                weights=weights, 
                loss_params={'n_bcd_mean': n_bcd_mean},
                maxiter=maxiter, 
                patience=patience,
                min_candidates=n_bcd_min,
                max_candidates=max_candidates, 
                mean_candidates=n_bcd_mean,
                history=history,
                propose_method=propose_method,
                initialize=initialize,
                verbose=verbose,
                n_repeats=n_repeats,
                )
            
            if history:
                record_spots_amps[i] = new_fit_vars
                record_bcd_amps[i] = new_amp_mean
                record_bcd_selection[i] = ssc_results['best_combi']
                
            if i < (n_steps_rescale_used - 1):
                # if not the last step, rescale amplitudes per bit from selected spots
                # we do not modify amplitude_std as we don't use that variable in the end
                select_spots = make_mask_from_used_spots(
                    len(coords),
                    spots_combinations,
                    ssc_results['best_combi'],
                    reverse=True,
                    )
                # modify spots amplitudes per round
                new_fit_vars = np.copy(fit_vars)
                for bit_idx in range(n_bits):
                    select_bit = spot_rounds == bit_idx
                    select = select_spots[select_bit]
                    if np.any(select):
                        new_fit_vars[select_bit] = transform_bit_data(new_fit_vars[select_bit], select, r_mean=1, reverse=True)
                    else:
                        if verbose > 0:
                            print(f'Skip modify spots amplitudes for bit {bit_idx} because empty selection')
                # recompute barcodes mean amplitude
                new_amp_mean = recompute_barcodes_amplitudes(new_fit_vars, spots_combinations, verbose=1)
                stats_norm.loc[:, 'amplitude'] = new_amp_mean

                # Recompute individual barcodes loss
                if verbose > 0:
                    print(f"Computing individual barcodes loss for iteration {i+1}")
                use_cols = [x for x in stats_norm.columns if x != 'loss']
                assert(len(use_cols) == len(weights))
                stats_norm = compute_individual_losses(stats_norm[use_cols], weights, inplace=True)
                if bcd_per_spot_params is not None:
                    stats_norm.loc[:, 'loss'] = stats_norm.loc[:, 'loss'] + loss_multiplicity
                else:
                    print("all steps performed")

    results = {
        'bcd_sequences': bcd_sequences[ssc_results['best_combi'], :],
        'stats': stats.loc[ssc_results['best_combi'], :],
        'rescaler': rescaler,
        'n_barcodes': np.sum(ssc_results['best_combi']),
    }
    if history:
        results['record_new_candidates'] = ssc_results['record_new_candidates']
        results['record_selection_test'] = ssc_results['record_selection_test']
        results['record_selection'] = ssc_results['record_selection']
        results['record_loss'] = ssc_results['record_loss']
        results['record_decision'] = ssc_results['record_decision']
        results['all_stats'] = stats
        results['stats_norm'] = stats_norm
        results['rescaled_bit_spots_amplitudes'] = record_spots_amps
        results['rescaled_bit_barcodes_amplitudes'] = record_bcd_amps
        results['rescaled_bit_bcd_selection'] = record_bcd_selection
    if return_contribs:
        results['spots_combinations'] = spots_combinations
        results['best_combi'] = ssc_results['best_combi']
    if return_barcodes_loss:
        results['barcodes_loss'] = stats_norm['loss']
        results['all_stats'] = stats
        results['best_combi'] = ssc_results['best_combi']
    if return_extra:
        results['pairs'] = pairs
        results['stats_norm'] = stats_norm
        results['spots_bcd'] = spots_bcd
        results['counts'] = counts
        results['n_bcd_min'] = n_bcd_min
        results['n_bcd_max'] = n_bcd_max
        results['n_bcd_mean'] = n_bcd_mean
        results['max_candidates'] = max_candidates

    return results


def decode_optimized_chunks(
    chunk_id, 
    dir_save, 
    coords, 
    coords_lim, 
    fitted_vars, 
    spot_rounds, 
    codebook, 
    dist_params,
    n_pos_bits=None, 
    n_bits=None, 
    size_bcd_min=None, 
    size_bcd_max=None, 
    err_corr_dist=1, 
    weights='auto', 
    bcd_per_spot_params=None,
    max_positive_bits=0.5,
    barcodes_exploration='stochastic', 
    n_bcd_min_coef=0.75,
    n_bcd_max_coef=2, 
    maxiter=200, 
    patience='maxiter', 
    max_candidates='auto',
    propose_method='single_step', # 'iter_bcd', 'from_off', 'from_all'
    initialize='maxloss', # 'meanloss', 'minloss', 'minrand', 'maxloss', 'maxrand'
    trim_network=True,
    file_exist='skip',
    extra_str='',
    verbose=0,
    ):
    """
    Example
    -------
    >>> spot_rounds = detected_coords.loc[select_coords, ['rounds']].values.ravel()
    """

    file_path = dir_save / f"decoded_localized_filtered_spots{extra_str}_chunk-{chunk_id}.csv"
    if file_path.exists():
        if file_exist == 'skip':
            return -1, -1
        elif file_exist == 'overwrite':
            print(f'{file_path} already exists and will be overwritten')
        elif file_exist == 'increment':
            i = 1
            file_path = dir_save / f"decoded_localized_filtered_spots_chunk-{chunk_id}-{i}.csv"
            while file_path.exists():
                i += 1
                file_path = dir_save / f"decoded_localized_filtered_spots_chunk-{chunk_id}-{i}.csv"
            
    # Select coordinates in chunk
    # we don't need to filter along z or x for one stripe
    conditions = []
    if coords_lim['z_lim_min'] is not None:
        conditions.append(coords[:, 0] > coords_lim['z_lim_min'])
    if coords_lim['z_lim_max'] is not None:
        conditions.append(coords[:, 0] < coords_lim['z_lim_max'])
    if coords_lim['y_lim_min'] is not None:
        conditions.append(coords[:, 1] > coords_lim['y_lim_min'])
    if coords_lim['y_lim_max'] is not None:
        conditions.append(coords[:, 1] < coords_lim['y_lim_max'])
    if coords_lim['x_lim_min'] is not None:
        conditions.append(coords[:, 2] > coords_lim['x_lim_min'])
    if coords_lim['x_lim_max'] is not None:
        conditions.append(coords[:, 2] < coords_lim['x_lim_max'])
    conditions = np.stack(conditions, axis=1)
    select_coords = np.logical_and.reduce(conditions, axis=1)
    n_tot = select_coords.sum()
    
    if n_tot > 0:
        # On selected spots
        spot_ids = np.arange(select_coords.sum())

        optim_results = optimize_spots(
            coords=coords[select_coords, :], 
            fit_vars=fitted_vars[select_coords].reshape((-1, 1)), 
            spot_ids=spot_ids, 
            spot_rounds=spot_rounds[select_coords], 
            dist_params=dist_params, 
            codebook=codebook,
            n_pos_bits=n_pos_bits, 
            n_bits=n_bits, 
            size_bcd_min=size_bcd_min, 
            size_bcd_max=size_bcd_max, 
            err_corr_dist=err_corr_dist, 
            weights=weights, 
            bcd_per_spot_params=bcd_per_spot_params,
            max_positive_bits=max_positive_bits,
            barcodes_exploration=barcodes_exploration,
            n_bcd_min_coef=n_bcd_min_coef,
            n_bcd_max_coef=n_bcd_max_coef, 
            maxiter=maxiter, 
            patience=patience, 
            max_candidates=max_candidates,
            propose_method=propose_method,
            initialize=initialize,
            trim_network=trim_network,
            history=False,
            return_extra=False,
            verbose=verbose,
            )

        # save results
        results_decoded = optim_results['stats']
        # Filter spots in overlap region
        # conditions on other borders are already fulfilled
        y_trim_max = coords_lim['y_trim_max']
        results_decoded = results_decoded.loc[results_decoded['y'] < y_trim_max, :]
    else:
        results_decoded = make_empty_stats(dist_params)
    results_decoded.to_csv(file_path, index=False)

    n_bcd = len(results_decoded)
    if verbose > 0:
        print(f"On chunk {chunk_id} decoded {n_bcd} barcodes from {n_tot} spots")
    return n_bcd, n_tot


def merge_stepwise_results(all_res, dict_idxs=None):
    """
    Merge dictionaries stored in a list with identical keys, concatenating
    1D and 2D arrays and numbers. Stores additional keys to inform about index
    of origin in the list.
    
    Parameters
    ----------
    all_res : list(dict)
        List of dictionnaries holding arrays and numbers.
    dict_idxs : list of array
        Indices to track the dictionary of origin of merged data.
    
    Returns
    -------
    merged_res : dict
        Dictionnary with same keys as those stored in the list, but with values
        merged by concatenation of arrays or numbers. Adds keys with `_step_idx`
        to allow filtering by dictionary of origin in the input list.
    """
    
    if len(all_res) == 1:
        return all_res[0]
    if dict_idxs is None:
        dict_idxs = range(len(all_res))
    
    # initialize dictionary of merged results across optimization steps
    # look for first dictionnary with non None values
    merged_res = {}
    start_key = 0
    while len(merged_res) == 0:
        if all_res[start_key] is not None:
            for key, val in all_res[start_key].items():
                if isinstance(val, (list, np.ndarray, pd.DataFrame)):
                    merged_res[key] = val
                    # inform on optimization step 
                    merged_res[key + '_step_idx'] = [start_key] * len(val)
                elif isinstance(val, Number):
                    merged_res[key] = [val]
                    # inform on optimization step 
                    merged_res[key + '_step_idx'] = [start_key]
        start_key += 1

    # add following steps' results
    if len(all_res) > start_key:
        for i, res in enumerate(all_res[start_key:]):
            if res is not None:
                for key, val in res.items():
                    if isinstance(val, np.ndarray):
                        if val.ndim == 1:
                            merged_res[key] = np.hstack((merged_res[key], val))
                        else:
                            merged_res[key] = np.vstack((merged_res[key], val))
                        merged_res[key + '_step_idx'] = merged_res[key + '_step_idx'] + [i+start_key] * len(val)
                    elif isinstance(val, list):
                        merged_res[key] = merged_res[key] + val
                        merged_res[key + '_step_idx'] = merged_res[key + '_step_idx'] + [i+start_key] * len(val)
                    elif isinstance(val, pd.DataFrame):
                        merged_res[key] = pd.concat([merged_res[key], val], axis=0, ignore_index=True)
                        merged_res[key + '_step_idx'] = merged_res[key + '_step_idx'] + [i+start_key] * len(val)
                    elif isinstance(val, Number):
                        merged_res[key] = merged_res[key] + [val]
                        merged_res[key + '_step_idx'] = merged_res[key + '_step_idx'] + [i+start_key]
                
    # convert lists of step ids into arrays for potential further indexing
    for key, val in merged_res.items():
        if isinstance(val, list):
            # avoid transforming list of other types of objects
            if isinstance(val[0], Number):
                merged_res[key] = np.array(val)
            
    return merged_res


def step_wise_optimize_decoding(
    coords, 
    fit_vars, 
    spot_ids, 
    spot_rounds, 
    search_distances,
    error_distances,
    bcd_per_spot_params,
    codebook,
    weights,
    history=False,
    return_extra=False,
    steps_filter_intensity=None,
    steps_filter_loss=None,
    filter_intensity_perc=5,
    filter_loss_perc=95,
    verbose=0,
    ):
    """
    Perform a step-wise optimization by varying at each step the search radius for
    spots neighbors and the allowed error in barcode sequence.
    
    Parameters
    ----------
    coords : ndarray
        Coordinates of localized spots.
    fit_vars : ndarray
        Results of 3D fits during spots localization.
    spot_ids : array
        Unique identifier for each spot.
    spot_rounds : array
        Bit index in barcode sequence for each spot.
    search_distances : array or list
        All distances successively considered.
    error_distances : integer or array or list
        Number of allowed errors in barcode sequence, if integer it applies
        to all steps, if iterable it applies per step.
    steps_filter_intensity : integer or array or list
        Steps at which barcodes are filtered given their mean amplitude.
    steps_filter_loss : integer or array or list
        Steps at which barcodes are filtered given their loss.
    """

    all_steps_results = []
    filter_intensity = None
    filter_loss = None
    return_barcodes_loss = steps_filter_loss is not None
    rescaler = None # at the first step a new one is created
    
    for step_idx, dist_params in enumerate(search_distances):
        if verbose > 1:
            print(f'Decoding step {step_idx}')
        
        if isinstance(error_distances, Number):
            err_corr_dist = error_distances
        else:
            err_corr_dist = error_distances[step_idx]      
        
        if bcd_per_spot_params is None:
            multiplicity_params = None
        elif isinstance(bcd_per_spot_params, dict) and 'w0' in bcd_per_spot_params.keys():
            multiplicity_params = bcd_per_spot_params
        else:
            multiplicity_params = bcd_per_spot_params[step_idx]         

        optim_results = optimize_spots(
            coords=coords, 
            fit_vars=fit_vars, 
            spot_ids=spot_ids, 
            spot_rounds=spot_rounds, 
            dist_params=dist_params, 
            err_corr_dist=err_corr_dist,
            bcd_per_spot_params=multiplicity_params,
            codebook=codebook,
            weights=weights,
            history=history, 
            return_extra=return_extra,
            propose_method='iter_bcd',
            n_repeats=1,
            return_contribs=True,
            return_barcodes_loss=return_barcodes_loss,
            filter_intensity=filter_intensity,
            filter_loss=filter_loss,
            rescaler=rescaler,
            )
        
        all_steps_results.append(optim_results)
        if verbose > 1:
            print(f"Step {step_idx}: Decoded {optim_results['n_barcodes']} barcodes.")
    
        if step_idx < len(search_distances) - 1:
            # update usable data from previous optimization round
            spots_mask = make_mask_from_used_spots(
                len(coords),
                optim_results['spots_combinations'],
                optim_results['best_combi'],
            )
            if spots_mask.sum() == 0:
                # No spots left
                if verbose > 1:
                    print(f'Step {step_idx}: There is no remaining spot')
                merged_results = merge_stepwise_results(all_steps_results)
                return merged_results
            
            # Compute hard filters threshold values if needed
            if steps_filter_intensity is not None:
                if isinstance(steps_filter_intensity, int) and step_idx+1 == steps_filter_intensity or \
                    isinstance(steps_filter_intensity, (list, np.ndarray)) and step_idx+1 in steps_filter_intensity:
                        data = [dico['stats']['amplitude'].values for dico in all_steps_results]
                        data = np.hstack(data)
                        filter_intensity = np.percentile(data, filter_intensity_perc)
                        if verbose > 2:
                            print(f'Filter intensity at step {step_idx+1} at {filter_intensity}')
            
            if steps_filter_loss is not None:
                if isinstance(steps_filter_loss, int) and step_idx+1 == steps_filter_loss or \
                    isinstance(steps_filter_loss, (list, np.ndarray)) and step_idx+1 in steps_filter_loss:
                        data = [dico['barcodes_loss'] for dico in all_steps_results]
                        data = np.hstack(data)
                        filter_loss = np.percentile(data, filter_loss_perc)
                        if verbose > 2:
                            print(f'Filter losses at step {step_idx+1} at {filter_loss}')
                        
            if step_idx == 0:
                # Save the data transformer to apply further identical transformations next steps
                # /!\ maybe implement another way to load the rescaler, in case there was no spot found during first step
                rescaler = all_steps_results[0]['rescaler']
            
            # discard spots and their stats for next decoding step
            coords = coords[spots_mask, :]
            fit_vars = fit_vars[spots_mask, :]
            # spot_ids = spot_ids[spots_mask]
            # not really usefull to keep track of specific spot ids
            # TODO: ↳ actually it can be, implement that in the future
            spot_ids = np.arange(len(coords))
            spot_rounds = spot_rounds[spots_mask]

    merged_results = merge_stepwise_results(all_steps_results)
    return merged_results


def remove_duplicate_pairs(pairs):
    """
    Remove redundant rows in a 2D array.
    
    Parameters
    ----------
    pairs : ndarray
        The (n_pairs x 2) array of neighbors indices.

    Returns
    -------
    uniq_pairs : ndarray
        Array of unique pairs, the content of each row is sorted.
    
    Example
    -------
    >>> pairs = [[4, 3],
                 [1, 2],
                 [3, 4],
                 [2, 1]]
    >>> remove_duplicate_pairs(pairs)
    array([[1, 2],
           [3, 4]])
    """
    
    uniq_pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return uniq_pairs


def build_rdn(coords, r, coords_ref=None, **kwargs):
    """
    Reconstruct edges between nodes by radial distance neighbors (rdn) method.
    An edge is drawn between each node and the nodes closer 
    than a threshold distance (within a radius).

    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    r : float, optional
        Radius in which nodes are connected.
    coords_ref : ndarray, optional
        Source points in the network, `pairs` will indicate edges from `coords_ref`
        to `coords`, if None, `coords` is used, the network is undirected.
    
    Examples
    --------
    >>> coords = make_simple_coords()
    >>> pairs = build_rdn(coords, r=60)

    Returns
    -------
    pairs : ndarray
        The (n_pairs x 2) matrix of neighbors indices.
    """
    
    tree = BallTree(coords, **kwargs)
    if coords_ref is None:
        ind = tree.query_radius(coords, r=r)
    else:
        ind = tree.query_radius(coords_ref, r=r)
    # clean arrays of neighbors from self referencing neighbors
    # and aggregate at the same time
    source_nodes = []
    target_nodes = []
    for i, arr in enumerate(ind):
        neigh = arr[arr != i]
        source_nodes.append([i]*(neigh.size))
        target_nodes.append(neigh)
    # flatten arrays of arrays
    source_nodes = np.fromiter(itertools.chain.from_iterable(source_nodes), int).reshape(-1,1)
    target_nodes = np.fromiter(itertools.chain.from_iterable(target_nodes), int).reshape(-1,1)
    # remove duplicate pairs
    pairs = np.hstack((source_nodes, target_nodes))
    if coords_ref is None:
        pairs = remove_duplicate_pairs(pairs)
    return pairs

def array_to_dict(arr):
    return dict(enumerate(arr))


def dict_to_array(dico):
    return np.array(list(dico.values()))

def dict_to_2D_array(dico):
    table = np.array(list(dico.items()))
    keys = table[:, 0]
    vals = np.vstack(table[:, 1])
    return keys, vals

def df_to_listarray(df, col_split, usecols=None):
    """
    Transform a dataframe into a list of 2D arrays, grouped by `col_split`.
    The reciprocal function is `list_arrays_to_df`.
    """
    if usecols is None:
        usecols = df.columns
    listarray = [
        df.loc[df[col_split] == i, usecols].values for i in np.unique(df[col_split])
    ]
    return listarray

def array_str_to_int(data):
    """
    Transform an array of strings into an array of integers.
    """
    nb_rows = len(data)
    nb_cols = len(data[0, 0])
    int_array = np.zeros((nb_rows, nb_cols), dtype=int)
    for i in range(nb_rows):
        int_array[i] = np.array([int(s) for s in data[i, 0]])
    return int_array

def list_arrays_to_df(data, data_col_name=None, index_col_name='round'):
    """
    Transform a list of 2D array to a dataframe, with an additional column 
    indicating the array index of each row.
    The reciprocal function is `df_to_listarray`.
    """
    stacked = np.vstack(data)
    ids= []
    for i, arr in enumerate(data):
        ids.extend([i] * len(arr))
    ids = np.array(ids).reshape((-1, 1))
    if data_col_name is None:
        if stacked.shape[1] == 2:
            data_col_name = ['y', 'x']
        elif stacked.shape[1] == 3:
            data_col_name = ['z', 'y', 'x']
    df = pd.DataFrame(data=np.hstack([stacked, ids]), columns=data_col_name + [index_col_name])
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    return df

def unique_nested_iterable(a):
    return functools.reduce(operator.iconcat, a, [])