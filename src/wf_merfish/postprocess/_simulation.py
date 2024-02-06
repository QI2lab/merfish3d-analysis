import numpy as np
import pandas as pd
import itertools
from numbers import Number
from sklearn.neighbors import BallTree

def generate_barcoded_spots(species, n_spots, tile_shape, codebook, noise=None, seed=0,
                            hybrid_probas=None, hybrid_targets=None,
                            mask=None, scale=None):
    """
    Parameters
    ----------
    species: list
        Name of species whose FISH spots are simulated.
    n_spots : int | array | list
        Number of spots per species, identical for all species if it's an int,
        otherwise specific to each species if list of array.
    tile_shape : array | list
        Image dimensions, can be any length but 2 or 3 make sense.
    codebook : dict
        Association from species to rounds barcodes, given by
        a string of zeros and ones like '01001101'.
    noise : float or list(float), default None
        If not None, uniform noise added to coordinates, different for the axial direction 
        if a list of 2 elements is provided.
    hybrid_probas : list(float), default None
        If not None, binding probabilities are modeled, each element of the list
        correspond to binding probabilities of each stage of hybridization (for amplified method). 
        For each molecule, the number of binding events varies only between rounds (imaging rounds),
        but this number can vary between molecules during previous steps.
    hybrid_targets : list(float), default None
        If not None, number of hybridization target sequences for each hybridization step. 
    mask : ndarray
        If not None, the surface or volume where spots can be present.
    scale : array
        Relate pixels of the mask to physical coordinates.
    
    Returns
    -------
    coords : DataFrame
        Coordinates of all spots with their identity.

    Example
    -------
    >>> species = ['a', 'b', 'c']
    >>> n_spots = 4
    >>> tile_shape = [25, 250, 500]
    >>> codebook = {'a': '01', 'b': '10', 'c': '11'}
    >>> generate_barcoded_spots(species, n_spots, tile_shape, codebook)

    Notes
    -----
    The last hybridization probabilities are generally higher than the first ones, 
    as imaging probes can diffuse and find their target faster than encoding probes.
    """

    nb_species = len(species)
    nb_dims = len(tile_shape)
    nb_rounds = len(next(iter(codebook.values())))
    if isinstance(n_spots, int):
        n_spots = [n_spots] * nb_species
    coords = []
    true_coords = []
    true_specs = []
    round_ids = []
    specs = []
    all_species_hybrid_probes = []

    if seed is not None:
        np.random.seed(seed)
    if mask is not None:
        mask_coords = np.where(mask)
        n_positive = len(mask_coords[0])
    
    for spec_id, (spec, n_spec) in enumerate(zip(species, n_spots)):
        barcode = codebook[spec]
        # go from '01001101' to [1, 4, 5, 7]
        spec_rounds = [i for i, x in enumerate(barcode) if x == '1']
        # number of positive rounds
        n_pos_round = sum([int(x) for x in barcode])
        # total number of spots across images
        n_spec_tot  = n_spec * n_pos_round

        # generate coordinates, identical across rounds
        if mask is None:
            spec_coords = [np.random.random(n_spec) * dim for dim in tile_shape]
            spec_coords = np.vstack(spec_coords).T
        else:
            spec_coords_idx = np.random.choice(np.arange(n_positive), size=n_spec, replace=True)
            spec_coords = [axis_coords[spec_coords_idx] for axis_coords in mask_coords]
            spec_coords = np.vstack(spec_coords).T
        if scale is not None:
            spec_coords = spec_coords* scale.reshape((1, -1))
        # save species coordinates to the list of unique ideal coordinates
        true_coords.append(spec_coords)
        # repeat and stack coordinates to match number of rounds
        spec_coords = np.tile(spec_coords, (n_pos_round, 1))
        coords.append(spec_coords)
        # indicate at what round coordinates are observed
        for round_id in spec_rounds:
            round_ids.extend([round_id] * n_spec)
        # indicate the ground truth species for all spots
        specs.extend([spec] * n_spec_tot)
        true_specs.extend([spec] * n_spec)

        # hybridization binding probabilities
        if hybrid_probas is not None:
            # add distribution of hybridization event for each hybrid step
            for hyb_id, hybrid_target in enumerate(hybrid_targets):
                # check if there are different targets per species
                # smaller mRNAs usually have fewer encoding probes for example
                if isinstance(hybrid_target, list):
                    n_targets = hybrid_target[spec_id]
                else:
                    n_targets = hybrid_target
                if hyb_id == 0:
                    # draw for each molecule the number of hybridized encoding probes
                    hybridized_probes = np.random.binomial(
                        n=n_targets, 
                        p=hybrid_probas[hyb_id], 
                        size=(n_spec, 1),
                        )
                    # repeat and stack to match the number of imaging rounds for this species
                    spec_hybridized_probes = np.tile(hybridized_probes, (n_pos_round, 1))
                else:
                    # draw for each molecule the number of hybridized encoding probes in new
                    # hybridization layer given the number of probes in the previous step
                    hybridized_probes = np.random.binomial(
                        # repeat experiment for each probe of previous step
                        n=spec_hybridized_probes[:, hyb_id - 1] * n_targets, 
                        p=hybrid_probas[hyb_id], 
                        size=n_spec_tot,
                        ).reshape(-1, 1)
                    spec_hybridized_probes = np.hstack((spec_hybridized_probes, hybridized_probes))
            all_species_hybrid_probes.append(spec_hybridized_probes)


    coords = np.vstack(coords)
    true_coords = np.vstack(true_coords)

    if noise is not None:
        if nb_dims == 2:
            add_noise = np.random.uniform(low=-noise, high=noise, size=coords.shape)
        elif nb_dims == 3:
            if isinstance(noise, Number):
                add_noise = np.random.uniform(low=-noise, high=noise, size=coords.shape)
            else:
                add_noise = np.hstack([
                    np.random.uniform(low=-noise[0], high=noise[0], size=(len(coords), 1)),
                    np.random.uniform(low=-noise[1], high=noise[1], size=(len(coords), 2))
                ])
        coords = coords + add_noise
        # clean coordinates outside image boundaries because of noise
        for i in range(nb_dims):
            coords[coords[:, i] < 0, i] = 0
            coords[coords[:, i] > tile_shape[i]-1, i] = tile_shape[i] - 1

    if nb_dims == 2:
        coords_names = ['y', 'x']
    elif nb_dims == 3:
        coords_names = ['z', 'y', 'x']
    else:
        coords_names = [f'dim-{x}' for x in range(nb_dims)]
    
    coords = pd.DataFrame(data=coords, columns=coords_names)
    true_coords = pd.DataFrame(data=true_coords, columns=coords_names)
    coords['rounds'] = round_ids
    coords['species'] = specs
    true_coords['species'] = true_specs

    # add hybridized probes
    if hybrid_probas is not None:
        all_species_hybrid_probes = np.vstack(all_species_hybrid_probes)
        col_names = [f'n_probes_{i}' for i in range(all_species_hybrid_probes.shape[1])]
        all_species_hybrid_probes = pd.DataFrame(data=all_species_hybrid_probes, columns=col_names)
        coords = pd.concat((coords, all_species_hybrid_probes), axis=1)

    if noise is None:
        return coords
    else:
        return coords, true_coords


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


def link_true_decoded_spots(src, trg, pairs=None, r=0.7,
                            col_spec='species', col_coords=['z', 'y', 'x']):
    """
    Link decoded spots to true spots.
    
    Parameters
    ----------
    src : dataframe
        Coordinates and species of true spots.
    trg : dataframe
        Coordinates and species of decoded spots.
    pairs : ndarray
        Links matching true and decoded spots by their ids, shape (n_links x 2).
    r : float
        Search radius for neighbors between true and decoded spots.
    col_spec : str
        Column used to compare spots' identities.
    col_coords : list(str)
        Columns used as spots coordinates.
        
    Returns
    -------
    linkage : dict
        Number of true positives (`perfect_matches`), incorrect matches , false 
        negatives (`no_match_true`) and false positives (`no_match_decoded`).
    
    Example
    -------
    >>> perfect_matches, incorrect_matches, no_match_true, no_match_decoded = \
        link_true_decoded_spots(true_coords, decoded)
    """
    
    if pairs is None:
        pairs = build_rdn(
            coords=trg[col_coords].values, 
            r=r, 
            coords_ref=src[col_coords].values,
            )
    
    perfect_matches = []
    incorrect_matches = []

    uniq_pairs_srcs = np.unique(pairs[:, 0])
    # need smart and flexible way to convert the species data
    src_species = src[col_spec].values
    trg_species = trg[col_spec].__array__()

    # First look for perfect matches only, from true to infered
    # perfect matches should be the same from infered to true

    # usable source spots to be liked with target spots
    use_spots_src = np.full(len(src), True)
    used_spots_trg = set()
    for src_idx in range(len(src)):
        if use_spots_src[src_idx]:
            src_spec = src_species[src_idx]
            select = pairs[:, 0] == src_idx
            trg_idxs = pairs[select, 1]
            # discard target spots already used
            trg_idxs = np.array(list(set(trg_idxs).difference(used_spots_trg)))
            if len(trg_idxs) > 0:
                trg_specs = trg_species[trg_idxs]
                # take the closest spot with matching species
                select_candidates =  trg_specs == src_spec
                if select_candidates.sum() > 0:
                    if select_candidates.sum() == 1:
                        trg_match_idx = trg_idxs[select_candidates][0]
                    else:
                        # if multiple matching species, look for the minimal distance
                        src_coords = src[col_coords].values[src_idx].reshape((1, -1))
                        trg_coords = trg[col_coords].values[trg_idxs[select_candidates]]
                        distances = np.linalg.norm(src_coords - trg_coords, axis=1)
                        trg_match_idx = trg_idxs[select_candidates][np.argmin(distances)]
                    if isinstance(trg_match_idx, np.ndarray):
                        print('src_idx:', src_idx)
                        print('trg_idxs:', trg_idxs)
                        print('distances:', distances)
                        print('trg_match_idx:', trg_match_idx)
                    perfect_matches.append([src_idx, trg_match_idx])
                    # upate usable source and used target spots
                    use_spots_src[src_idx] = False
                    used_spots_trg.add(trg_match_idx)

    # Then link closest spots, even with species mismatch
    for src_idx in range(len(src)):
        if use_spots_src[src_idx]:
            select = pairs[:, 0] == src_idx
            trg_idxs = pairs[select, 1]
            # discard target spots already used
            trg_idxs = np.array(list(set(trg_idxs).difference(used_spots_trg)))
            if len(trg_idxs) > 0:
                # take the closest spot, regardless of species
                    if len(trg_idxs) == 1:
                        trg_match_idx = trg_idxs[0]
                    else:
                        # if multiple matching species, look for the minimal distance
                        src_coords = src[col_coords].values[src_idx].reshape((1, -1))
                        trg_coords = trg[col_coords].values[trg_idxs]
                        distances = np.linalg.norm(src_coords - trg_coords, axis=1)
                        trg_match_idx = trg_idxs[np.argmin(distances)]
                    incorrect_matches.append([src_idx, trg_match_idx])
                    # upate usable source and used target spots
                    use_spots_src[src_idx] = False
                    used_spots_trg.add(trg_match_idx)
    # reshape in case of one of the arrays is empty, we can still index on it
    perfect_matches = np.array(perfect_matches).reshape((-1, 2))
    incorrect_matches = np.array(incorrect_matches).reshape((-1, 2))

    all_match_true = set(perfect_matches[:, 0]).union(set(incorrect_matches[:, 0]))
    no_match_true = set(range(len(src))).difference(all_match_true)
    no_match_true = np.array(list(no_match_true))
    all_match_decoded = set(perfect_matches[:, 1]).union(set(incorrect_matches[:, 1]))
    no_match_decoded = set(range(len(trg))).difference(all_match_decoded)
    no_match_decoded = np.array(list(no_match_decoded))
    
    linkage = {
        'perfect_matches': perfect_matches,
        'incorrect_matches': incorrect_matches,
        'no_match_true': no_match_true,
        'no_match_decoded': no_match_decoded,
    }
    return linkage


def make_linkage_stats(linkage, verbose=1, return_stats=True, as_dict=False):
    """
    Compute statistics given the links between true and decoded spots.
    
    Example
    -------
    >>> linkage = link_true_decoded_spots(true_coords, decoded_coords, pairs)
    >>> n_TP, n_FP, n_FN, n_incorrect, accuracy = make_linkage_stats(linkage)
    """
    
    n_TP = len(linkage['perfect_matches'])
    n_FP = len(linkage['no_match_decoded'])
    n_FN = len(linkage['no_match_true'])
    n_incorrect = len(linkage['incorrect_matches'])
    accuracy = n_TP / (n_TP + n_FP + n_FN + n_incorrect)

    if verbose > 0:
        print("TP:", n_TP)
        print("FP:", n_FP)
        print("FN:", n_FN)
        print("incorrect:", n_incorrect)
        print(f"accuracy: {accuracy:.3f}")
    if return_stats:
        if as_dict:
            stats = {
                'n_TP': n_TP, 
                'n_FP': n_FP, 
                'n_FN': n_FN, 
                'n_incorrect': n_incorrect, 
                'accuracy': accuracy,
            }
        else:
            stats = n_TP, n_FP, n_FN, n_incorrect, accuracy
        return stats