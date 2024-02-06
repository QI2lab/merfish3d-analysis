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
from typing import Optional, Any, List, Tuple, Union, Iterable, Callable, Dict, Set
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TO DO: - what is the plan with the stepwise decoding?
#        - how to rework for a better dask strategy?

def make_mask_from_used_spots(n_spots: int, 
                              combinations: List[Tuple[int, ...]], 
                              select: List[bool], 
                              reverse: bool = False) -> np.ndarray:
    """
    Make a boolean mask to filter spots used in combinations.

    Parameters:
    -----------
    n_spots : int
        The total number of spots.

    combinations : List[Tuple[int, ...]]
        List of combinations of spot indices.

    select : List[bool]
        A list of boolean values indicating whether each combination is selected.

    reverse : bool, optional
        If True, the function returns the inverse of the mask (spots not used).
        Default is False.

    Returns:
    --------
    np.ndarray
        A boolean mask indicating which spots are used based on the given combinations and selection.

    Example:
    --------
    >>> combinations = optim_results['spots_combinations']
    >>> select = optim_results['best_combi']
    >>> mask = make_mask_from_used_spots(len(coords), combinations, select)
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
        source: np.ndarray,
        target: np.ndarray,
        dist_method: str = "xy_z_orthog",
        metric: str = "euclidean",
        tilt_vector: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute distances between sets of points.

    Parameters
    ----------
    source : np.ndarray
        Coordinates of the first set of points.

    target : np.ndarray
        Coordinates of the second set of points.

    dist_method : str, optional
        Method used to compute distances. If 'isotropic', standard distances are computed considering all axes
        simultaneously. If 'xy_z_orthog', two distances are computed for the xy plane and along the z axis
        respectively. If 'xy_z_tilted', two distances are computed for the tilted plane and its normal axis.
        Default is 'xy_z_orthog'.

    metric : str, optional
        The distance metric to be used. Default is 'euclidean'.

    tilt_vector : np.ndarray, optional
        Tilt vector used in the 'xy_z_tilted' method. Default is None.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        The computed distances. If 'dist_method' is 'isotropic', a single array is returned.
        If 'dist_method' is 'xy_z_orthog', a tuple of two arrays is returned (dist_z, dist_xy).
        If 'dist_method' is 'xy_z_tilted', the method is not implemented yet, and NotImplementedError is raised.

    Example
    -------
    >>> source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> target = np.array([[0, 0, 0], [-3, 0, 2], [0, 0, 10]])
    >>> compute_distances(source, target)
    array([0., 4., 0.])

    >>> compute_distances(source, target, metric='L1')
    array([0, 2, 7])
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
    source: np.ndarray,
    target: np.ndarray,
    dist_params: Union[float, np.ndarray],
    dist_method: str = "isotropic",
    metric: str = "euclidean",
    return_bool: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    For each spot in a given round ("source"), find if there are neighbors
    in another round ("target") within a given distance.

    Parameters
    ----------
    source : np.ndarray
        Coordinates of spots in the source round.

    target : np.ndarray
        Coordinates of spots in the target round.

    dist_method : str, optional
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog. Default is "isotropic".

    dist_params : float or np.ndarray
        Threshold distance to classify spots as neighbors.
        Multiple thresholds can be used depending on the method.
        Typically, 2 thresholds for xy plane and z axis in xy_z_orthog.
        
    return_bool : bool, optional
        If True, return a vector indicating the presence of neighbors
        for spots in the source set. Default is False.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Pairs of neighbors if return_bool is False.
        Array indicating the presence of neighbors for each spot in their source round if return_bool is True.

    Example
    -------
    >>> source = np.array([[0, 0, 0], [0, 2, 0]])
    >>> target = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    >>> find_neighbor_spots_in_round(source, target, dist_params=2)
    array([[0, 2]])

    >>> find_neighbor_spots_in_round(source, target, dist_params=[1, 1], return_bool=True)
    array([ True, False])
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


def make_neighbors_lists(
    coords: np.ndarray,
    spot_ids: np.ndarray,
    spot_rounds: np.ndarray,
    dist_params: Union[float, np.ndarray],
    max_positive_bits: int,
    dist_method: str = "isotropic",
    verbose: int = 1,
    **kwargs
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Build a list of neighbors' ids for each round, starting from each spot.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of spots.

    spot_ids : np.ndarray
        Spot IDs.

    spot_rounds : np.ndarray
        Round information for each spot.

    dist_params : float or np.ndarray
        Threshold distance to classify spots as neighbors.
        Multiple thresholds can be used depending on the method.

    max_positive_bits : int
        Maximum number of positive bits in the resulting combinations.

    dist_method : str, optional
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog. Default is "isotropic".

    verbose : int, optional
        If greater than 0, display progress using tqdm. Default is 1.

    Returns
    -------
    List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
        A list of tuples representing neighbors' ids for each round.

    Example
    -------
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


def spots_product(
    *args: List[int],
    repeat: int = 1,
    verbose: int = 1
) -> List[List[Optional[int]]]:
    """
    Make combinations of elements, picking either a single element or none per list.
    Absence of element is indicated by -1. This is a modified recipe from itertools.

    Parameters
    ----------
    *args : List[int]
        Lists of elements.

    repeat : int, optional
        Number of times to repeat the lists. Default is 1.

    verbose : int, optional
        If greater than 1, display progress using tqdm. Default is 1.

    Yields
    ------
    List[Optional[int]]
        Combinations of elements, where absence of element is indicated by -1.

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


def make_combinations_single_spot(
    spots: List[List[Optional[int]]],
    size_bcd_min: int = 2,
    size_bcd_max: int = 6
) -> List[List[Optional[int]]]:
    """
    Build barcodes from all possible combinations of localized spots' ids across rounds.

    Parameters
    ----------
    spots : List[List[Optional[int]]]
        Lists of neighbors in each round for each spot.

    size_bcd_min : int, optional
        Minimum size of the barcode. Default is 2.

    size_bcd_max : int, optional
        Maximum size of the barcode. Default is 6.

    Returns
    -------
    List[List[Optional[int]]]
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


def make_combinations(
    neighbors: List[List[List[Optional[int]]]],
    size_bcd_min: int = 2,
    size_bcd_max: int = 6,
    verbose: int = 1
) -> List[List[List[Optional[int]]]]:
    """
    Build barcodes from all possible combinations of localized spots' ids across rounds.

    Parameters
    ----------
    neighbors : List[List[List[Optional[int]]]]
        Lists of neighbors in each round for each spot.

    size_bcd_min : int, optional
        Minimum size of the barcode. Default is 2.

    size_bcd_max : int, optional
        Maximum size of the barcode. Default is 6.

    verbose : int, optional
        If greater than 0, display progress using tqdm. Default is 1.

    Returns
    -------
    List[List[List[Optional[int]]]]
        List of barcode sequences.

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
    # remove duplicate sets of combinations
    combinations = list(set(combinations))
    # TODO: test optimisation of the previous line with the next 2 lines:
    # combinations = list(set(tuple(map(tuple, c)) for c in combinations))
    # combinations = [list(map(list, c)) for c in combinations]

    return combinations


def make_barcode_from_contributors(
    contribs: List[List[int]],
    spot_rounds: np.ndarray,
    n_bits: Optional[int] = None,
    verbose: int = 1
) -> Tuple[np.ndarray, Dict[int, set]]:
    """
    Generate barcode sequences from contributors.

    Parameters
    ----------
    contribs : List[List[int]]
        IDs of spots contributing to barcode sequences.

    spot_rounds : np.ndarray
        Round index of each spot.

    n_bits : Optional[int], optional
        Number of bits in barcodes. If not provided, inferred from the maximum round index.

    verbose : int, optional
        If greater than 0, display progress using tqdm. Default is 1.

    Returns
    -------
    Tuple[np.ndarray, Dict[int, set]]
        Tuple containing:
        - bcd_sequences : np.ndarray
            Barcode sequences from all possible spots IDs combinations.
        - spots_bcd : Dict[int, set]
            Information for each spot about the barcode IDs it contributes to.

    Example
    -------
    >>> contribs = [(0, 5), (1, 3), (2, 4, 6)]
    >>> spot_rounds = np.array([0, 0, 1, 1, 2, 3, 3])
    >>> make_barcode_from_contributors(contribs, spot_rounds, n_bits=None)
    (array([[1, 0, 0, 1],
           [1, 1, 0, 0],
           [0, 1, 1, 1]]), {0: {0}, 5: {0}, 1: {1}, 3: {1}, 2: {2}, 4: {2}, 6: {2}})
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
                # TODO: check if new spot_id are added or replace previous ones
                spots_bcd[spot_id].update({i})  # it' a set, not a dictionary
            else:
                spots_bcd[spot_id] = {i}

    return bcd_sequences, spots_bcd


def std_distance(coords: np.ndarray) -> float:
    """
    Compute the standard deviation of distances of points to their mean center.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of points.

    Returns
    -------
    float
        Standard deviation of distances to the mean center.

    Example
    -------
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> std_distance(coords)
    0.4714045207910317
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


def find_barcode_min_dist(
    barcode: np.ndarray,
    codebook_keys: np.ndarray,
    codebook_vals: np.ndarray,
    max_err: int = 1
) -> Tuple[Optional[str], int]:
    """
    Find the minimum distance between a barcode and all barcodes in a codebook,
    and infer the identity of this barcode.

    Parameters
    ----------
    barcode : np.ndarray
        Barcode detected starting from spots in all rounds, shape (n_barcodes, n_rounds).

    codebook_keys : np.ndarray
        List of keys of the codebook.

    codebook_vals : np.ndarray
        2D array of binary values of the codebook.

    max_err : int, optional
        Maximum allowable error for inferring barcode identity. Default is 1.

    Returns
    -------
    Tuple[Optional[str], int]
        A tuple containing:
        - bcd_species : Optional[str]
            Inferred barcode identity.
        - min_err : int
            Minimum Hamming distance between the given barcode and the codebook.

    Example
    -------
    >>> barcode = np.array([1, 1, 1, 0, 0, 0, 0, 0]).reshape((1, -1))
    >>> codebook = {'a': np.array([1, 1, 1, 1, 0, 0, 0, 0]),
                    'b': np.array([0, 0, 1, 1, 1, 1, 0, 0]),
                    'c': np.array([0, 0, 0, 0, 1, 1, 1, 1])}
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


def compute_errors(
    barcodes: np.ndarray,
    codebook_vals: np.ndarray
) -> np.ndarray:
    """
    Compute the Hamming distance between barcode sequences and their closest
    allowed sequence in a codebook.

    Parameters
    ----------
    barcodes : np.ndarray
        Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).

    codebook_vals : np.ndarray
        2D array of binary values of the codebook.

    Returns
    -------
    np.ndarray
        Array containing the Hamming distance for each barcode.

    Example
    -------
    >>> barcodes = np.array([[0, 1, 1, 1, 1, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0, 0, 0],
                             [1, 1, 0, 0, 0, 1, 1, 1]])
    >>> codebook = {'a': np.array([1, 1, 1, 1, 0, 0, 0, 0]),
                    'b': np.array([0, 0, 1, 1, 1, 1, 0, 0]),
                    'c': np.array([0, 0, 0, 0, 1, 1, 1, 1])}
    >>> cbk_keys, codebook_vals = dict_to_2D_array(codebook)
    >>> compute_errors(barcodes, codebook_vals)
    array([2, 1, 3])
    """

    # compute distances between barcode and codebook's barcodes
    errors = cdist(barcodes, codebook_vals, metric='cityblock').astype(int)
    # get the minimun errors / Hamming distance
    min_err = errors.min(axis=1)

    return min_err


def build_barcodes_stats(
    coords: np.ndarray,
    fit_vars: np.ndarray,
    contribs: List[List[int]],
    barcodes: np.ndarray,
    codebook_keys: np.ndarray,
    codebook_vals: np.ndarray,
    dist_method: str = "isotropic",
    max_err: int = 1,
    verbose: int = 1
) -> pd.DataFrame:
    """
    Make the matrix that holds all information about all potential barcodes,
    their contributing spots ids, and derived statistics.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of localized spots.

    fit_vars : np.ndarray
        Results of 3D fits during spots localization. For now the columns are:
        [amplitude,].

    contribs : List[List[int]]
        List of contributing spot ids for each barcode.

    barcodes : np.ndarray
        Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).

    codebook_keys : np.ndarray
        List of keys of the codebook.

    codebook_vals : np.ndarray
        2D array of binary values of the codebook.

    dist_method : str, optional
        Method used to compute distances. 'isotropic' or 'xy_z_orthog'. Default is 'isotropic'.

    max_err : int, optional
        Maximum allowable error for inferring barcode identity. Default is 1.

    verbose : int, optional
        Verbosity level. Default is 1.

    Returns
    -------
    stats : pd.DataFrame
            Statistics of barcodes: mean position [z, y, x], (z dispersion), x/y(/z) dispersion,
            mean amplitude, sequences min error to codebook sequences, barcodes' species inferred 
            from the smallest Hamming distance between sequences.

    Example
    -------
    >>> coords = np.array([[0, 0, 0],
                           [0, 1, 1],
                           [3, 2, 2],])
    >>> fit_vars = np.array([4, 4, 10]).reshape((-1, 1))
    >>> contribs = [(0, 1), (0, 1, 2)]
    >>> barcodes = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0, 0, 0]])
    >>> codebook = {'a': np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                    'b': np.array([1, 1, 1, 1, 0, 0, 0, 0]),
                    'c': np.array([1, 0, 1, 0, 1, 0, 1, 0])}
    >>> cbk_keys, cbk_vals = dict_to_2D_array(codebook)
    >>> build_barcodes_stats(coords, fit_vars, contribs, barcodes, cbk_keys, cbk_vals)
    (       z    y    x  z/x/y std  amplitude  amplitude std  error species
    0  0.0  0.5  0.5   1.414214   4.0          2.0     0      b
    1  1.0  1.0  1.0   1.414214   6.0          1.0     0      b)
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


def transform_bit_data(
    data: np.ndarray,
    select: Optional[np.ndarray] = None,
    r_mean: float = 1,
    reverse: bool = False
) -> np.ndarray:
    """
    Transform amplitudes so that selected spots have a specific mean amplitude,
    and reverse the min and max of the distribution.

    Parameters
    ----------
    data : np.ndarray
        Array of amplitudes.

    select : Optional[np.ndarray], optional
        Indices of the selected spots. If None, the entire array is considered. Default is None.

    r_mean : float, optional
        Target mean of selected spots' amplitudes. Default is 1.

    reverse : bool, optional
        If True, reverse the distribution. Default is False.

    Returns
    -------
    np.ndarray
        Transformed amplitudes.

    Example
    -------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> transform_bit_data(data, select=[1, 3], r_mean=2, reverse=True)
    array([4., 2., 3., 0., 5.])
    """
    # TODO: check this generated example
    
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


def make_empty_stats(dist_params: Union[Number, np.ndarray]) -> pd.DataFrame:
    """
    Make an empty stats DataFrame in case no barcode exists.

    Parameters
    ----------
    dist_params : Union[Number, np.ndarray]
        The distance parameters. If a single number is provided, assume isotropic distances.
        If an array is provided, assume anisotropic distances.

    Returns
    -------
    pd.DataFrame
        An empty DataFrame with columns for barcode statistics.

    Example
    -------
    >>> make_empty_stats(1.0)
    Empty DataFrame
    Columns: [z, y, x, z/x/y std, amplitude, amplitude std, error, species]
    Index: []

    >>> make_empty_stats(np.array([1.0, 2.0, 3.0]))
    Empty DataFrame
    Columns: [z, y, x, z std, x/y std, amplitude, amplitude std, error, species]
    Index: []
    """
    
    if isinstance(dist_params, Number):
        colnames = ['z', 'y', 'x', 'z/x/y std', 'amplitude', 'amplitude std', 'error', 'species']
    else:
        colnames = ['z', 'y', 'x', 'z std', 'x/y std', 'amplitude', 'amplitude std', 'error', 'species']
    stats = pd.DataFrame(np.full((0, len(colnames)), None), columns=colnames)
    return stats


def reverse_sigmoid(x: np.ndarray, w0: float, b: float) -> np.ndarray:
    """
    Return the reverse sigmoid of an array.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    w0 : float
        Parameter controlling the asymptote of the reverse sigmoid curve.

    b : float
        Parameter controlling the shape of the reverse sigmoid curve.

    Returns
    -------
    np.ndarray
        Reverse sigmoid values for the input array.

    Example
    -------
    >>> x_values = np.array([1, 2, 3, 4])
    >>> reverse_sigmoid(x_values, w0=5.0, b=2.0)
    array([4.94117647, 4.5       , 4.16494845, 4.05882353])
    """
    return w0 - x**4 / (b**4 + x**4)

def logistic(x: np.ndarray, w0: float = 1.0, k: float = 1.0, b: float = 0.0) -> np.ndarray:
    """
    Compute the logistic function for an array.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    w0 : float, optional
        Asymptotic value of the logistic curve. Default is 1.0.

    k : float, optional
        Steepness of the logistic curve. Default is 1.0.

    b : float, optional
        Horizontal shift of the logistic curve. Default is 0.0.

    Returns
    -------
    np.ndarray
        Logistic values for the input array.

    Example
    -------
    >>> x_values = np.array([1, 2, 3, 4])
    >>> logistic(x_values, w0=2.0, k=2.0, b=1.0)
    array([1.        , 1.76159416, 1.96402758, 1.99505475])
    """
    return w0 / (1 + np.exp(-k * (x - b)))

def compute_barcode_multiplicity(
    spots_combinations: List[List[int]],
    spots_bcd: Dict[int, List[int]],
    bcd_per_spot_params: Dict[str, Union[float, int]],
    fct: callable = None
) -> np.ndarray:
    """
    Compute a loss for each barcode depending on the multiplicity of their
    contributing spots, which is the number of potential barcodes per spot.
    A reverse sigmoid function is applied to the distribution of the maximum
    multiplicities of barcodes to help "ranking" barcodes.

    Parameters
    ----------
    spots_combinations : List[List[int]]
        List of lists representing the combinations of spots for each barcode.

    spots_bcd : Dict[int, List[int]]
        Dictionary where keys are spot indices and values are lists of barcode indices
        indicating which barcodes the spots contribute to.

    bcd_per_spot_params : Dict[str, Union[float, int]]
        Parameters to compute the multiplicity loss. Keys are 'w0' and 'b', and 
        optionally 'weight' to modify its importance relative to other losses.
        The default weight is ~1/3 so this loss has the same importance as spots 
        dispersion and mean amplitude when added to the barcode loss.

    fct : callable, optional
        Function to compute the multiplicity value from a list of spot indices.
        The default is np.max.

    Returns
    -------
    np.ndarray
        Multiplicity loss values for each barcode.

    Example
    -------
    >>> spots_combinations = [[0, 1, 2, 3], [2, 3, 4, 5], [6, 7, 8, 9]]
    >>> spots_bcd = {0: [0], 1: [0], 2: [0, 1], 3: [0, 1], 4: [1], 5: [1], 6: [2], 7: [2], 8: [2], 9: [2]}
    >>> bcd_per_spot_params = {'w0': 1.0, 'b': 1.0, 'weight': 0.5}
    >>> compute_barcode_multiplicity(spots_combinations, spots_bcd, bcd_per_spot_params)
    array([0.36552929, 0.36552929, 0.25      ])
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


def remap_spots_bcd(
    spots_bcd: Dict[int, List[int]],
    select: np.ndarray,
    size: int = None,
    verbose: int = 1
) -> Dict[int, List[int]]:
    """
    Modify the values of a dictionary given a boolean array that filters this
    dictionary.

    Parameters
    ----------
    spots_bcd : Dict[int, List[int]]
        For each spot, all barcodes it contributed to.

    select : np.ndarray
        Filters the list of spots combinations making the barcodes.

    size : int, optional, default None
        Provide a known count of selected elements to avoid computing it.

    verbose : int, optional, default 1
        Verbosity level. If 0, silent; if 1, show progress bar; if >1, show details.

    Returns
    -------
    Dict[int, List[int]]
        Modified dictionary with remapped values.

    Example
    -------
    If we start with:
    >>> spots_combinations = [(0, 1, 2), (1, 2), (0, 1, 3), (4, 5), (3, 6)]
    >>> spots_bcd = {0: [0, 2], 1: [0, 1, 2], 2: [0, 1], 3:[2, 4], 4: [3], 5: [3], 6: [4]}
    and we filter `spots_combinations` with a boolean array
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


# def filter_barcodes(
#     sequences: np.ndarray,
#     stats: np.ndarray,
#     combinations: List[Tuple[int]],
#     spots_bcd: Optional[Dict[int, List[int]]] = None,
#     max_err: int = 1,
#     err_col: int = -1,
#     verbose: int = 1
# ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int]], Optional[Dict[int, List[int]]], np.ndarray]:
#     """
#     Filter the barcodes statistics array given barcodes sequences errors.

#     Parameters
#     ----------
#     sequences : np.ndarray
#         Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).

#     stats : np.ndarray
#         Statistics of barcodes: mean position [z, y, x], (z dispersion), x/y dispersion,
#         mean amplitude, sequences min error to codebook sequences.

#     combinations : List[Tuple[int]]
#         List of contributing spot ids for each barcode.

#     spots_bcd : Optional[Dict[int, List[int]]], default None
#         For each spot, all barcodes it contributed to.

#     max_err : int, optional, default 1
#         Maximum allowed error in barcode sequences.

#     err_col : int, optional, default -1
#         Index of the error column in the stats array.

#     verbose : int, optional, default 1
#         Verbosity level. If 0, silent; if 1, show progress bar; if >1, show details.

#     Returns
#     -------
#     Tuple[np.ndarray, np.ndarray, List[Tuple[int]], Optional[Dict[int, List[int]]], np.ndarray]
#         Filtered sequences, stats, combinations, remapped spots_bcd, and selection array.

#     Example
#     -------
#     >>> sequences = np.array([[0, 1, 1, 1, 1, 0, 0, 0],
#                              [1, 1, 1, 0, 0, 0, 0, 0]])
#     >>> stats = np.array([[0.0, 0.5, 0.5, 1.41421356, 1.15470054, 4.0, 2.0, 1],
#                           [1.0, 1.0, 1.0, 1.41421356, 1.15470054, 6.0, 1.0, 1]])
#     >>> combinations = [(0,), (1,)]
#     >>> spots_bcd = {0: [0, 1], 1: [0, 1]}
#     >>> filter_barcodes(sequences, stats, combinations, spots_bcd, max_err=1, err_col=-1)
#     (array([[1, 1, 1, 0, 0, 0, 0, 0]]),
#     array([[1., 1., 1., 1.41421356, 1.15470054, 6., 1., 1.]]),
#     [(1,)],
#     {0: [0], 1: [0]},
#     array([False,  True]))
#     """
#     # TODO: check generated example
    
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


# def filter_max_bcd_per_spot(
#     max_bcd_per_spot: int,
#     bcd_sequences: np.ndarray,
#     spots_combinations: List[Tuple[int]],
#     spots_bcd: Dict[int, List[int]],
#     verbose: int = 0
# ) -> Tuple[np.ndarray, List[Tuple[int]], Dict[int, List[int]]]:
#     """
#     Filter barcodes and spots when the latter have too many related barcodes.

#     Parameters
#     ----------
#     max_bcd_per_spot : int
#         Maximum number of barcodes spots can be related to before being filtered.

#     bcd_sequences : np.ndarray
#         Barcode sequences from all possible spots ids combinations.

#     spots_combinations : List[Tuple[int]]
#         For each barcode, its contributing spots.

#     spots_bcd : Dict[int, List[int]]
#         For each spot, all barcodes it contributed to.

#     verbose : int, optional, default 0
#         Verbosity level. If 0, silent; if >0, show details.

#     Returns
#     -------
#     Tuple[np.ndarray, List[Tuple[int]], Dict[int, List[int]]]
#         Filtered bcd_sequences, spots_combinations, and spots_bcd.

#     Example
#     -------
#     >>> max_bcd_per_spot = 1
#     >>> bcd_sequences = np.array([[1, 0, 0],
#                                   [0, 1, 1],
#                                   [1, 0, 1]])
#     >>> spots_combinations = [(0,), (1, 2), (0, 2)]
#     >>> spots_bcd = {0: [0, 2], 1: [1], 2: [1, 2]}
#     >>> filter_max_bcd_per_spot(max_bcd_per_spot, bcd_sequences, spots_combinations, spots_bcd)
#     (array([[1, 0, 1]]), [(0, 2)], {0: [0]})
#     """

#     # TODO: check generated example
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


def prefilter_barcodes_error(
    bcd_sequences: np.ndarray,
    codebook_vals: np.ndarray,
    spots_combinations: List[Tuple[int]],
    spots_bcd: Optional[Dict[int, List[int]]] = None,
    max_err: int = 1,
    verbose: int = 1
) -> Tuple[np.ndarray, List[Tuple[int]], Optional[Dict[int, List[int]]], np.ndarray, np.ndarray]:
    """
    Filter the arrays of barcodes, contributing spots, and update the 
    spot-barcode network given barcode sequence errors.

    Parameters
    ----------
    bcd_sequences : np.ndarray
        Barcode sequences from all possible spots ids combinations.

    codebook_vals : np.ndarray
        2D array of binary values of the codebook.

    spots_combinations : List[Tuple[int]]
        For each barcode, its contributing spots.

    spots_bcd : Optional[Dict[int, List[int]]], default None
        For each spot, all barcodes it contributed to.

    max_err : int, optional, default 1
        Maximum allowed error in barcode sequences.

    verbose : int, optional, default 1
        Verbosity level. If 0, silent; if >0, show details.

    Returns
    -------
    Tuple[np.ndarray, List[Tuple[int]], Optional[Dict[int, List[int]]], np.ndarray, np.ndarray]
        Filtered bcd_sequences, spots_combinations, new_spots_bcd, errors, and select.

    Example
    -------
    >>> bcd_sequences = np.array([[1, 0, 0],
                                  [0, 1, 1],
                                  [1, 0, 1]])
    >>> codebook_vals = np.array([[1, 0, 0],
                                  [0, 1, 1],
                                  [1, 0, 1]])
    >>> spots_combinations = [(0,), (1, 2), (0, 2)]
    >>> spots_bcd = {0: [0, 2], 1: [1], 2: [1, 2]}
    >>> prefilter_barcodes_error(bcd_sequences, codebook_vals, spots_combinations, spots_bcd, max_err=1)
    (array([[1, 0, 0],
            [1, 0, 1]]), [(0,), (0, 2)], {0: [0], 2: [1]})
    """
    
    # TODO: check generated example
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


def filter_barcodes_array(
    data: np.ndarray,
    threshold: float,
    bcd_sequences: np.ndarray,
    spots_combinations: List[Tuple[int]],
    stats: pd.DataFrame,
    direction: str = 'greater',
    spots_bcd: Optional[Dict[int, List[int]]] = None,
    stats_norm: Optional[pd.DataFrame] = None,
    verbose: int = 1
) -> List[Union[np.ndarray, List[Tuple[int]], pd.DataFrame, Optional[Dict[int, List[int]]], Optional[pd.DataFrame], np.ndarray]]:
    """
    Filter barcodes and several related variables and update the 
    spot-barcode network given an array of values and a hard threshold.

    Parameters
    ----------
    data : np.ndarray
        Array of values used for filtering.

    threshold : float
        Hard threshold for filtering.

    bcd_sequences : np.ndarray
        Barcode sequences from all possible spots ids combinations.

    spots_combinations : List[Tuple[int]]
        For each barcode, its contributing spots.

    stats : pd.DataFrame
        Statistics of barcodes.

    direction : str, optional, default 'greater'
        Direction of the filtering. 'greater' keeps values greater than the threshold,
        'less' keeps values less than the threshold.

    spots_bcd : Optional[Dict[int, List[int]]], default None
        For each spot, all barcodes it contributed to.

    stats_norm : Optional[pd.DataFrame], default None
        Additional statistics used for normalization.

    verbose : int, optional, default 1
        Verbosity level. If 0, silent; if >0, show details.

    Returns
    -------
    List[Union[np.ndarray, List[Tuple[int]], pd.DataFrame, Optional[Dict[int, List[int]]], Optional[pd.DataFrame], np.ndarray]]
        Filtered bcd_sequences, spots_combinations, stats, new_spots_bcd, stats_norm, and select.

    Example
    -------
    >>> data = np.array([1.2, 0.8, 1.5, 0.7, 2.0])
    >>> threshold = 1.0
    >>> bcd_sequences = np.array([[1, 0, 0],
                                  [0, 1, 1],
                                  [1, 0, 1],
                                  [0, 0, 1],
                                  [1, 1, 1]])
    >>> spots_combinations = [(0,), (1, 2), (0, 2), (2,), (0, 1, 2)]
    >>> stats = pd.DataFrame({
           'z': [0, 1, 2, 1, 0],
           'y': [1, 2, 3, 2, 1],
           'x': [0, 1, 0, 2, 1],
           'error': [1, 0, 1, 0, 2],
           'species': ['a', 'b', 'c', 'a', 'b']
        })
    >>> filter_barcodes_array(data, threshold, bcd_sequences, spots_combinations, stats)
    [array([[0, 1, 1],
           [1, 0, 1],
           [1, 1, 1]]), [(0, 2), (0, 1, 2), (0, 1, 2)], 
           'z': [1, 2, 0],
           'y': [2, 3, 1],
           'x': [1, 0, 1],
           'error': [0, 1, 2],
           'species': ['b', 'c', 'b']], 
           {0: [1, 2], 1: [0, 1, 2], 2: [0, 1, 2]}, None, array([ True, False,  True, False,  True])]
    """
    
    # TODO: check generated example
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


def build_barcodes_network(spots_combinations: List[List[int]], spots_bcd: Dict[int, List[int]], verbose: int = 1) -> Dict[int, Set[int]]:
    """
    Build the graph of barcodes, where nodes are barcode ids and edges represent 
    common contributing spot ids.

    Parameters
    ----------
    spots_combinations : List[List[int]]
        For each barcode, its contributing spots.
    spots_bcd : Dict[int, List[int]]
        For each spot, all barcodes it contributed to.
    verbose : int, optional, default 1
        Verbosity level. If 0, silent; if >0, show details.

    Returns
    -------
    Dict[int, Set[int]]
        Dictionary representing the edges in the barcode network.

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
    bcd_sequences: np.ndarray,
    spots_combinations: List[List[int]],
    spots_bcd: Dict[int, List[int]],
    stats: pd.DataFrame,
    stats_norm: pd.DataFrame,
    verbose: int = 1,
) -> Union[None, Tuple[None, np.ndarray, List[List[int]], pd.DataFrame, pd.DataFrame]]:
    """
    Look for connected barcodes, and discard the connected ones with the worst loss.
    Update related barcodes objects.

    Parameters
    ----------
    bcd_sequences : np.ndarray
        Barcode sequences from all possible spots ids combinations.
    spots_combinations : List[List[int]]
        For each barcode, its contributing spots.
    spots_bcd : Dict[int, List[int]]
        For each spot, all barcodes it contributed to.
    stats : pd.DataFrame
        Statistics of barcodes: mean position [z, y, x], (z dispersion), x/y dispersion,
        mean amplitude, sequences min error to codebook sequences.
    stats_norm : pd.DataFrame
        Normalized statistics of barcodes.
    verbose : int, optional, default 1
        Verbosity level. If 0, silent; if >0, show details.

    Returns
    -------
    Union[None, Tuple[None, np.ndarray, List[List[int]], pd.DataFrame, pd.DataFrame]]
        Tuple containing None for `pairs` (to signal further functions that barcodes have been filtered),
        filtered `bcd_sequences`, updated `spots_combinations`, and filtered `stats` and `stats_norm`.

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
    A class to escale values between 0 and 1 for each column using a low and high
    percentile bound for each column.
    If percentile range is given for each colums, it must be as a list or array 
    if data is an array, or as a dictionnary too if data is a dataframe, 
    with keys corresponding to the dataframe's columns.

    Methods
    -------
    find_boundaries(X, low, up, col_name=None, y=None):
        Find the boundaries for rescaling based on percentiles.
    fit(X, y=None):
        Fit the rescaler to the input data.

    Attributes
    ----------
    data_type : str
        The type of the input data ('ndarray' or 'dataframe').
    lower_bound : list or dict
        Lower bounds for rescaling.
    upper_bound : list or dict
        Upper bounds for rescaling.
    
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
        
    def find_boundaries(self, X: np.ndarray, low: float, up: float,
                        col_name: Optional[Union[str, int]] = None, y: Optional[np.ndarray] = None) -> None:
        """
        Find the boundaries for rescaling based on percentiles.

        Parameters
        ----------
        X : np.ndarray
            Input data array.
        low : float
            Lower percentile for rescaling.
        up : float
            Upper percentile for rescaling.
        col_name : Union[str, int], optional
            Column name or index (only applicable if data_type is 'dataframe').
        y : np.ndarray, optional
            Target array (not used in this method).

        Returns
        -------
        None

        Notes
        -----
        - This method updates the lower_bound and upper_bound attributes based on the specified percentiles.
        """

        X = np.copy(X)
        thresh_low = np.percentile(X, low)
        thresh_up = np.percentile(X, up) - thresh_low
        if self.data_type == 'ndarray':
            self.lower_bound.append(thresh_low)
            self.upper_bound.append(thresh_up)  
        elif self.data_type == 'dataframe':
            self.lower_bound[col_name] = thresh_low
            self.upper_bound[col_name] = thresh_up  

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> 'PercentileRescaler':
        """
        Fit the rescaler to the input data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Input data array or DataFrame.
        y : np.ndarray, optional
            Target array (not used in this method).

        Returns
        -------
        PercentileRescaler
            The fitted rescaler instance.

        Notes
        -----
        - This method determines the type of the input data and initializes the rescaling parameters.
        """

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
    
    def transform(self, 
                  X: Union[np.ndarray, pd.DataFrame], 
                  y: Optional[np.ndarray] = None
                  ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform the input data based on the fitted rescaler.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Input data array or DataFrame.
        y : np.ndarray, optional
            Target array (not used in this method).

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            Transformed data.

        Notes
        -----
        - This method rescales each column of the input data based on the previously fitted rescaler parameters.
        """

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
    

def normalize_stats(
    stats: pd.DataFrame,
    rescaler: Optional[PercentileRescaler] = None,
    rescaler_kwargs: Optional[Dict[str, List[int]]] = None,
    reverse_cols: Optional[List[str]] = ['amplitude']
) -> Tuple[pd.DataFrame, Optional[PercentileRescaler]]:
    """
    Perform data transformation on barcode statistics to make variables more
    comparable to each other, enabling the use of more meaningful weights.

    Parameters
    ----------
    stats : pd.DataFrame
        DataFrame containing statistics such as (z dispersion), x/y dispersion, mean amplitude, std amplitude,
        sequence error. If coordinates (z, y, x) are present, they are discarded.
    rescaler : Optional[PercentileRescaler], default=None
        Rescaler object for transforming the data. If not provided, a new one will be created.
    rescaler_kwargs : Optional[Dict[str, List[int]]], default=None
        Keyword arguments to initialize the PercentileRescaler if `rescaler` is not provided.
        Default is set based on the presence of 'z std' in the columns.
    reverse_cols : Optional[List[str]], default=['amplitude']
        List of columns to reverse (0 <--> 1) in the resulting DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[PercentileRescaler]]
        A tuple containing the normalized statistics DataFrame and the rescaler used for the transformation.

    Notes
    -----
    - If 'z' is present in the columns, the coordinates 'z', 'y', 'x' are discarded in the normalized DataFrame.
    - If `rescaler` is not provided, a PercentileRescaler will be created with default or specified `rescaler_kwargs`.
    - If `reverse_cols` is provided, specified columns will be reversed (0 <--> 1) in the resulting DataFrame.

    Examples
    --------
    >>> normalized_stats, scaler = normalize_stats(stats_df)
    >>> normalized_stats, scaler = normalize_stats(stats_df, rescaler_kwargs={'perc_low': [0, 0, 0, 0], 'perc_up': [100, 100, 100, 100]})
    >>> normalized_stats, scaler = normalize_stats(stats_df, rescaler=my_custom_rescaler, reverse_cols=['amplitude', 'other_column'])
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


def compute_individual_losses(
    stats: pd.DataFrame,
    weights: Union[list, tuple, pd.Series, np.ndarray],
    inplace: bool = False
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Compute the contribution to loss of each individual barcode.

    Parameters
    ----------
    stats : pd.DataFrame
        DataFrame containing statistics such as (z dispersion), x/y dispersion, mean amplitude, std amplitude,
        sequence error, and species. There is no mean_z, mean_y, or mean_x.
    weights : Union[list, tuple, pd.Series, np.ndarray]
        Coefficients of each variable in stats for the loss. The last coefficient is for the number of barcodes
        in the combination considered.
    inplace : bool, default=False
        If True, stack the loss to the stats array and return stats.

    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        If inplace is True, returns the modified DataFrame with an additional 'loss' column.
        If inplace is False, returns the computed losses as a 1D array.

    Notes
    -----
    - The 'weights' array should include coefficients for each variable in 'stats' plus one additional coefficient
      for the number of barcodes in the combination considered.
    - If 'inplace' is True, the 'loss' column is added to the 'stats' DataFrame.

    Examples
    --------
    >>> losses = compute_individual_losses(stats_df, weights_array)
    >>> updated_stats = compute_individual_losses(stats_df, weights_array, inplace=True)
    """

    num_cols = [x for x in stats if x != 'species']
    loss = stats.loc[:, num_cols].values @ weights[:-1].reshape((-1, 1))
    if inplace:
        stats['loss'] = loss
        return stats
    return loss


def compute_selection_loss(
    indiv_loss: np.ndarray,
    weights: Union[list, tuple, np.ndarray],
    loss_params: Dict[str, Union[int, float]],
    fct_aggreg: Callable[[np.ndarray], float] = np.mean
) -> float:
    """
    Compute the selection loss based on individual barcode losses.

    Parameters
    ----------
    indiv_loss : np.ndarray
        Individual losses of barcodes.
    weights : Union[list, tuple, np.ndarray]
        Coefficients of each variable in stats for the loss. The last coefficient
        is for the number of barcodes in the combination considered.
    loss_params : Dict
        General information and parameters to parameterize the loss.
    fct_aggreg : Callable[[np.ndarray], float], default=np.mean
        Aggregation function to combine individual losses.

    Returns
    -------
    float
        The computed selection loss.
    """
    
    # Contribution of selection size to the loss
    loss_size = np.abs( (loss_params['n_bcd_mean'] - len(indiv_loss)) / loss_params['n_bcd_mean'] )
    loss = fct_aggreg(indiv_loss) + loss_size * weights[-1]
    return loss


def powerset(
    iterable: Iterable,
    size_min: int = 1,
    size_max: int = None
) -> Iterable[Tuple]:
    """
    Generate the powerset of an iterable.

    Parameters
    ----------
    iterable : Iterable
        The input iterable for which the powerset is generated.
    size_min : int, optional
        The minimum size of subsets in the powerset (default is 1).
    size_max : int, optional
        The maximum size of subsets in the powerset (default is the length of the iterable).

    Returns
    -------
    Iterable[Tuple]
        An iterable containing tuples representing subsets of the input iterable.

    Examples
    --------
    >>> list(powerset([1, 2, 3]))
    [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    >>> list(powerset([1, 2, 3], size_min=2)
    [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """

    if size_max is None:
        size_max = len(iterable)
    return itertools.chain.from_iterable(itertools.combinations(iterable, r) for r in range(size_min, size_max+1))


def is_valid(bcd_combination: Tuple[int, ...], 
             pairs: Dict[int, set], 
             n_bcd_min: int = 1, 
             n_bcd_max: Union[int, float] = float('inf')) -> bool:
    """
    Check if a combination of selected barcodes is valid, i.e. barcode ids are not
    connected in the barcode network due to common contributing spots, and the
    number of selected barcode is within a size range.

    Parameters
    ----------
    bcd_combination : Tuple[int, ...]
        The combination of selected barcode ids to be checked for validity.
    pairs : Dict[int, set]
        The barcode network represented as a dictionary of connected barcode ids.
    n_bcd_min : int, optional
        The minimum number of selected barcodes allowed (default is 1).
    n_bcd_max : Union[int, float], optional
        The maximum number of selected barcodes allowed (default is infinity).

    Returns
    -------
    bool
        True if the combination is valid, False otherwise.

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

    if not n_bcd_min <= len(bcd_combination) <= n_bcd_max:
        return False
    for i, bcd_src in enumerate(bcd_combination):
        if bcd_src in pairs.keys():
            for bcd_trg in pairs[bcd_src]:
                if bcd_trg in bcd_combination:
                    return False
    return True


def find_best_barcodes_combination(
    stats: pd.DataFrame,
    pairs: Dict[int, set],
    loss_function: Callable,
    weights: np.ndarray,
    loss_params: Optional[Dict[str, Union[int, float]]] = None,
    n_best: int = 1,
    n_bcd_min: int = 1,
    n_bcd_max: Optional[int] = None,
    n_bcd_combis: Optional[int] = None,
    verbose: int = 1,
) -> Tuple[np.ndarray, float]:
    """
    Try all combinations of barcodes and save the one with the best loss function.

    Parameters
    ----------
    stats : pd.DataFrame
        DataFrame containing barcode statistics.
    pairs : Dict[int, set]
        The barcode network represented as a dictionary of connected barcode ids.
    loss_function : Callable
        The loss function to be minimized, taking a Series of losses, weights, and optional parameters.
    weights : np.ndarray
        Coefficients of each variable in stats for the loss.
    loss_params : Optional[Dict[str, Union[int, float]]], optional
        General informations and parameters to parametrize the loss.
    n_best : int, optional
        The number of best combinations to save (default is 1).
    n_bcd_min : int, optional
        The minimum number of selected barcodes allowed (default is 1).
    n_bcd_max : Optional[int], optional
        The maximum number of selected barcodes allowed (default is None).
    n_bcd_combis : Optional[int], optional
        Total number of barcode combinations if known, used for progress tracking (default is None).
    verbose : int, optional
        Verbosity level (default is 1).

    Returns
    -------
    Tuple[np.ndarray, float]
        Tuple containing the best barcode combination and its corresponding loss.
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


def clean_selection(
    bcd_select: np.ndarray,
    pairs: Dict[int, set],
    new_candidates: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Correct a selection of barcodes by eliminating barcodes linked to each other
    in their network (`pairs`), while potentially iterating in a specific order.

    Parameters
    ----------
    bcd_select : np.ndarray
        Selected barcodes.
    pairs : Dict[int, set]
        Dictionary linking barcode ids to sets of neighboring barcode ids.
    new_candidates : Optional[np.ndarray], optional
        If not None, specific order to iterate over barcodes and eliminate their 
        neighbors (given by `pairs`) in the selection.

    Returns
    -------
    bcd_select : np.ndarray
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


def search_stochastic_combinations(
    stats: pd.DataFrame,
    pairs: Dict[int, set],
    loss_function: Callable,
    weights: np.ndarray,
    loss_params: Optional[Dict] = None,
    maxiter: int = 200,
    patience: Union[int, str] = 'maxiter',
    min_candidates: Optional[int] = None,
    max_candidates: Optional[Union[int, float]] = None,
    mean_candidates: Optional[float] = None,
    history: bool = False,
    initialize: str = 'maxloss',
    propose_method: str = 'single_step',
    n_repeats: int = 1,
    verbose: int = 1
) -> Dict[str, Union[np.ndarray, List[np.ndarray], pd.DataFrame]]:
    """
    Search the best set of barcodes with random combinations.

    Parameters
    ----------
    stats : pd.DataFrame
        DataFrame containing barcode statistics.
    pairs : Dict[int, set]
        Links pairs of barcodes by their id.
    loss_function : function
        Function to compute the loss.
    weights : np.ndarray
        Coefficients of each variable in stats for the loss.
    loss_params : Optional[Dict], optional
        General informations and parameters to parametrize the loss.
    maxiter : int, optional
        Maximum number of iterations. Default is 200.
    patience : Union[int, str], optional
        Patience parameter to control early stopping. If 'maxiter', patience is set to maxiter.
    min_candidates : Optional[int], optional
        Minimum number of candidates for each iteration. Default is None.
    max_candidates : Optional[Union[int, float]], optional
        Maximum number of candidates for each iteration. Default is None.
    mean_candidates : Optional[float], optional
        Mean number of candidates for each iteration. Default is None.
    history : bool, optional
        Whether to record the history of the optimization. Default is False.
    initialize : str, optional
        Method to initialize barcode candidates. Default is 'maxloss'.
    propose_method : str, optional
        Method to propose new candidates. Default is 'single_step'.
    n_repeats : int, optional
        Number of repeats for proposing new candidates. Default is 1.
    verbose : int, optional
        Verbosity level. Default is 1.

    Returns
    -------
    results : dict
        A dictionary containing the results of the optimization.
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
    

def optimize_spots(
    coords: pd.DataFrame,
    fit_vars: np.ndarray,
    spot_ids: np.ndarray,
    spot_rounds: np.ndarray,
    dist_params: Union[Number, Dict],
    codebook: Dict[str, str],
    n_pos_bits: Optional[int] = None,
    n_bits: Optional[int] = None,
    size_bcd_min: Optional[int] = None,
    size_bcd_max: Optional[int] = None,
    err_corr_dist: int = 1,
    weights: Union[str, np.ndarray] = 'auto',
    max_positive_bits: Union[int, float] = 0.5,
    barcodes_exploration: str = 'stochastic',
    n_bcd_min_coef: float = 0.75,
    n_bcd_max_coef: float = 2,
    maxiter: int = 200,
    patience: Union[int, str] = 'maxiter',
    max_candidates: Union[int, str] = 'auto',
    n_repeats: int = 1,
    filter_intensity: Optional[float] = None,
    filter_loss: Optional[float] = None,
    max_bcd_per_spot: Optional[int] = None,
    bcd_per_spot_params: Optional[Dict] = None,
    rescale_used_spots: bool = True,
    n_steps_rescale_used: int = 10,
    rescaler: Optional[callable] = None,
    rescaler_kwargs: Optional[Dict] = None,
    propose_method: str = 'single_step',
    initialize: str = 'maxloss',
    trim_network: bool = True,
    history: bool = False,
    return_extra: bool = False,
    return_contribs: bool = False,
    return_barcodes_loss: bool = False,
    verbose: int = 1
) -> Dict:
    """
    Optimize barcodes decoding based on spots data.

    Parameters
    ----------
    coords : pd.DataFrame
        DataFrame containing the coordinates of spots.
    fit_vars : np.ndarray
        Array containing fit variables.
    spot_ids : np.ndarray
        Array containing spot IDs.
    spot_rounds : np.ndarray
        Array containing spot rounds.
    dist_params : Union[Number, Dict]
        Distribution parameters for spot proximity.
    codebook : Dict[str, str]
        Mapping from gene names to binary sequences.
    n_pos_bits : Optional[int], optional
        Number of positive bits in the barcode sequence.
    n_bits : Optional[int], optional
        Total number of bits in the barcode sequence.
    size_bcd_min : Optional[int], optional
        Minimum size of a valid barcode selection.
    size_bcd_max : Optional[int], optional
        Maximum size of a valid barcode selection.
    err_corr_dist : int, optional
        Error correction distance for barcodes.
    weights : Union[str, np.ndarray], optional
        Weights for individual barcodes during optimization.
    max_positive_bits : Union[int, float], optional
        Maximum number or ratio of allowed positive bits in neighbors.
    barcodes_exploration : str, optional
        Exploration method for optimizing barcodes ('all', 'stochastic').
    n_bcd_min_coef : float, optional
        Coefficient for computing the minimum number of barcodes.
    n_bcd_max_coef : float, optional
        Coefficient for computing the maximum number of barcodes.
    maxiter : int, optional
        Maximum number of optimization iterations.
    patience : Union[int, str], optional
        Maximum number of non-improving iterations or 'maxiter'.
    max_candidates : Union[int, str], optional
        Maximum number of candidates for optimization.
    n_repeats : int, optional
        Number of repeats for stochastic optimization.
    filter_intensity : Optional[float], optional
        Threshold for filtering barcodes based on intensity.
    filter_loss : Optional[float], optional
        Threshold for filtering barcodes based on loss.
    max_bcd_per_spot : Optional[int], optional
        Maximum number of barcodes per spot.
    bcd_per_spot_params : Optional[Dict], optional
        Parameters for computing individual barcode multiplicity loss.
    rescale_used_spots : bool, optional
        If True, spots amplitudes are iteratively rescaled given the
        amplitude of spots used to build barcodes.
    n_steps_rescale_used : int, optional
        Number of iterations to rescale spots amplitudes.
    rescaler : Optional[callable], optional
        Custom rescaling function for amplitudes.
    rescaler_kwargs : Optional[Dict], optional
        Additional arguments for the rescaler function.
    propose_method : str, optional
        Method for proposing candidates during optimization.
    initialize : str, optional
        Initialization method for candidates.
    trim_network : bool, optional
        If True, trim the barcode network during optimization.
    history : bool, optional
        If True, record the optimization history.
    return_extra : bool, optional
        If True, return additional information in the result.
    return_contribs : bool, optional
        If True, return contributing spots information in the result.
    return_barcodes_loss : bool, optional
        If True, return barcodes loss information in the result.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    results : Dict
        Variable containing all results.
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
    if isinstance(weights, str) and weights == 'auto':
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
    chunk_id: int, 
    dir_save: Path, 
    coords: np.ndarray, 
    coords_lim: Dict[str, Optional[float]], 
    fitted_vars: np.ndarray, 
    spot_rounds: np.ndarray, 
    codebook: Dict[str, str], 
    dist_params: Union[float, Dict],
    n_pos_bits: Optional[int] = None, 
    n_bits: Optional[int] = None, 
    size_bcd_min: Optional[int] = None, 
    size_bcd_max: Optional[int] = None, 
    err_corr_dist: int = 1, 
    weights: Union[str, np.ndarray] = 'auto', 
    bcd_per_spot_params: Optional[Dict] = None,
    max_positive_bits: Union[int, float] = 0.5,
    barcodes_exploration: str = 'stochastic', 
    n_bcd_min_coef: float = 0.75,
    n_bcd_max_coef: float = 2, 
    maxiter: int = 200, 
    patience: Union[int, str] = 'maxiter', 
    max_candidates: Union[int, str] = 'auto',
    propose_method: str = 'single_step',
    initialize: str = 'maxloss',
    trim_network: bool = True,
    file_exist: str = 'skip',
    extra_str: str = '',
    verbose: int = 0,
) -> Tuple[int, int]:
    """
    Optimize barcodes decoding on spatially chunked data.
    
    Parameters
    ----------
    chunk_id : int
        ID of the chunk.
    dir_save : Path
        Path to the directory where results will be saved.
    coords : np.ndarray
        Array containing spot coordinates.
    coords_lim : Dict[str, Optional[float]]
        Dictionary with limits for spot coordinates (z_lim_min, z_lim_max, y_lim_min, y_lim_max, x_lim_min, x_lim_max).
    fitted_vars : np.ndarray
        Array containing fitted variables.
    spot_rounds : np.ndarray
        Array containing spot rounds.
    codebook : Dict[str, str]
        Mapping from gene names to binary sequences.
    dist_params : Union[float, Dict]
        Distribution parameters for spot proximity.
    n_pos_bits : Optional[int], optional
        Number of positive bits in the barcode sequence.
    n_bits : Optional[int], optional
        Total number of bits in the barcode sequence.
    size_bcd_min : Optional[int], optional
        Minimum size of a valid barcode selection.
    size_bcd_max : Optional[int], optional
        Maximum size of a valid barcode selection.
    err_corr_dist : int, optional
        Error correction distance for barcodes.
    weights : Union[str, np.ndarray], optional
        Weights for individual barcodes during optimization.
    bcd_per_spot_params : Optional[Dict], optional
        Parameters for computing individual barcode multiplicity loss.
    max_positive_bits : Union[int, float], optional
        Maximum number or ratio of allowed positive bits in neighbors.
    barcodes_exploration : str, optional
        Exploration method for optimizing barcodes ('all', 'stochastic').
    n_bcd_min_coef : float, optional
        Coefficient for computing the minimum number of barcodes.
    n_bcd_max_coef : float, optional
        Coefficient for computing the maximum number of barcodes.
    maxiter : int, optional
        Maximum number of optimization iterations.
    patience : Union[int, str], optional
        Maximum number of non-improving iterations or 'maxiter'.
    max_candidates : Union[int, str], optional
        Maximum number of candidates for optimization.
    propose_method : str, optional
        Method for proposing candidates during optimization.
    initialize : str, optional
        Initialization method for candidates.
    trim_network : bool, optional
        If True, trim the barcode network during optimization.
    file_exist : str, optional
        Handling of existing result files ('skip', 'overwrite', 'increment').
    extra_str : str, optional
        Extra string to append to the result file name.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    n_bcd, n_tot : Tuple[int, int]
        Number of decoded barcodes and total number of spots.

    Notes
    -----
    - The main results are saved as csv files in dir_save.

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


def merge_stepwise_results(
        all_res: List[Dict], 
        dict_idxs: Optional[List[Union[int, np.ndarray]]] = None) -> Dict:
    """
    Merge dictionaries stored in a list with identical keys, concatenating
    1D and 2D arrays and numbers. Stores additional keys to inform about index
    of origin in the list.

    Parameters
    ----------
    all_res : List[Dict]
        List of dictionaries holding arrays and numbers.
    dict_idxs : Optional[List[Union[int, np.ndarray]]], optional
        Indices to track the dictionary of origin of merged data.

    Returns
    -------
    merged_res : Dict
        Dictionary with the same keys as those stored in the list, but with values
        merged by concatenation of arrays or numbers. Adds keys with `_step_idx`
        to allow filtering by the dictionary of origin in the input list.
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
    coords: np.ndarray, 
    fit_vars: np.ndarray, 
    spot_ids: np.ndarray, 
    spot_rounds: np.ndarray, 
    search_distances: Union[int, List[Union[float, np.ndarray]]],
    error_distances: Union[int, List[Union[float, np.ndarray]]],
    bcd_per_spot_params: Optional[Union[Dict, List[Union[Dict, None]]]],
    codebook: np.ndarray,
    weights: str,
    history: bool = False,
    return_extra: bool = False,
    steps_filter_intensity: Optional[Union[int, List[int]]] = None,
    steps_filter_loss: Optional[Union[int, List[int]]] = None,
    filter_intensity_perc: int = 5,
    filter_loss_perc: int = 95,
    verbose: int = 0,
) -> Dict:
    """
    Perform a step-wise optimization by varying at each step the search radius for
    spots neighbors and the allowed error in barcode sequence.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of localized spots.
    fit_vars : np.ndarray
        Results of 3D fits during spots localization.
    spot_ids : np.ndarray
        Unique identifier for each spot.
    spot_rounds : np.ndarray
        Bit index in barcode sequence for each spot.
    search_distances : Union[int, List[Union[float, np.ndarray]]]
        All distances successively considered.
    error_distances : Union[int, List[Union[float, np.ndarray]]]
        Number of allowed errors in barcode sequence, if integer it applies
        to all steps, if iterable it applies per step.
    bcd_per_spot_params : Optional[Union[Dict, List[Union[Dict, None]]]]
        Parameters for barcode decoding per spot or per step.
    codebook : np.ndarray
        Codebook for decoding barcodes.
    weights : str
        Weighting scheme for optimization.
    history : bool, optional
        Whether to store optimization history.
    return_extra : bool, optional
        Whether to return extra information.
    steps_filter_intensity : Optional[Union[int, List[int]]], optional
        Steps at which barcodes are filtered given their mean amplitude.
    steps_filter_loss : Optional[Union[int, List[int]]], optional
        Steps at which barcodes are filtered given their loss.
    filter_intensity_perc : int, optional
        Percentage value for filtering barcodes by intensity.
    filter_loss_perc : int, optional
        Percentage value for filtering barcodes by loss.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    merged_results : Dict
        Merged dictionary with the results of the step-wise optimization.
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


def array_to_dict(arr: Union[np.ndarray, List]) -> Dict[int, Any]:
    """
    Convert an array into a dictionary with indices as keys.

    Parameters
    ----------
    arr : Union[np.ndarray, List]
        The input array or list.

    Returns
    -------
    Dict[int, Any]
        A dictionary where keys are the indices of the list elements, and
        values are the corresponding elements.
    """
    return dict(enumerate(arr))


def dict_to_array(dico: Dict[Any, Any]) -> np.ndarray:
    """
    Convert a dictionary's values into a NumPy array.

    Parameters
    ----------
    dico : Dict[Any, Any]
        The input dictionary.

    Returns
    -------
    np.ndarray
        A NumPy array containing the values from the input dictionary.
    """
    return np.array(list(dico.values()))


def dict_to_2D_array(dico: Dict[Any, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a dictionary of arrays to a 2D NumPy array.

    Parameters
    ----------
    dico : Dict[Any, np.ndarray]
        The input dictionary where values are NumPy arrays.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - The first array represents keys.
        - The second array is a 2D array containing values.

    Example
    -------
    >>> codebook = {'a': np.array([1, 1, 1, 1, 0, 0, 0, 0]),
                    'b': np.array([0, 0, 1, 1, 1, 1, 0, 0]),
                    'c': np.array([0, 0, 0, 0, 1, 1, 1, 1])}
    >>> cbk_keys, cbk_vals = dict_to_2D_array(codebook)
    """
    table = np.array(list(dico.items()))
    keys = table[:, 0]
    vals = np.vstack(table[:, 1])
    return keys, vals


def df_to_listarray(df: pd.DataFrame, col_split: str, usecols: Optional[Union[str, List[str]]] = None) -> List[np.ndarray]:
    """
    Transform a DataFrame into a list of 2D arrays, grouped by `col_split`.
    The reciprocal function is `list_arrays_to_df`.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col_split : str
        The column by which the DataFrame is split into groups.
    usecols : Union[str, List[str]], optional
        Columns to be included in the arrays, by default None (uses all columns).

    Returns
    -------
    listarray : List[np.ndarray]
        A list of 2D NumPy arrays, each corresponding to a group in `col_split`.
    """
    if usecols is None:
        usecols = df.columns
    listarray = [
        df.loc[df[col_split] == i, usecols].values for i in np.unique(df[col_split])
    ]
    return listarray


def array_str_to_int(data: np.ndarray) -> np.ndarray:
    """
    Transform an array of strings into an array of integers.

    Parameters
    ----------
    data : np.ndarray
        The input array of strings.

    Returns
    -------
    int_array : np.ndarray
        An array of integers obtained by converting the input strings.

    Notes
    -----
    The input array should be of shape (nb_rows, 1, nb_cols).
    """
    nb_rows = len(data)
    nb_cols = len(data[0, 0])
    int_array = np.zeros((nb_rows, nb_cols), dtype=int)
    for i in range(nb_rows):
        int_array[i] = np.array([int(s) for s in data[i, 0]])
    return int_array


def list_arrays_to_df(
        data: List[np.ndarray], 
        data_col_name: Union[None, List[str]] = None, 
        index_col_name: str = 'round') -> pd.DataFrame:
    """
    Transform a list of 2D arrays to a DataFrame, with an additional column 
    indicating the array index of each row.
    The reciprocal function is `df_to_listarray`.

    Parameters
    ----------
    data : List[np.ndarray]
        The list of 2D arrays to be transformed into a DataFrame.
    data_col_name : Union[None, List[str]], optional
        The column names for the data columns. If None, default names will be used.
    index_col_name : str, optional
        The name for the column indicating the array index of each row.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the data from the list of 2D arrays.
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


def unique_nested_iterable(a: Iterable[Iterable]) -> List:
    """
    Flatten a nested iterable and return a list of unique elements.

    Parameters
    ----------
    a : Iterable[Iterable]
        The nested iterable to be flattened.

    Returns
    -------
    List
        A list containing unique elements from the nested iterable.
    """
    # TODO: compare with generated optimization to ensure uniqueness:
    # return list(set(functools.reduce(operator.iconcat, a, [])))
    return functools.reduce(operator.iconcat, a, [])