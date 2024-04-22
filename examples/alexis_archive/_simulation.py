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


def add_drift_per_round(coords, magnitudes, directions, random_mags=True, 
                        col_rounds='rounds', col_coords=['z', 'y', 'x'],
                        return_shifts=False):
    """
    Add a common shift to spots positions for each round.
    
    Parameters
    ----------
    coords : dataframe
        Spots coordinates with z/y/x and rounds variables.
    magnitudes : float or list or array
        Length of displacement of spots per round.
    directions : list(tuple) or list(None)
        Directional constrains if tuple, or no constrain is None for each round.
    random_mags : bool
        If True, shifts magnitudes are drawn from a uniform distribution between
        0 and the element of `magnitudes` of each round, else it is this element.
    return_shifts : bool
        If True, shifts for all rounds are returned.
        
    Returns
    -------
    coords_shifted : dataframe
        Spots coordinates that have been shifted
    """
    
    coords_shifted = coords.copy()
    all_shifts = {}
    
    sorted_rounds = np.sort(coords[col_rounds].unique())        
    for r_idx in sorted_rounds:
        select = coords[col_rounds] == r_idx
        data = coords.loc[select, col_coords].values
        
        if isinstance(magnitudes, (list, np.ndarray, dict)):
            magnitude = magnitudes[r_idx]
        else:
            magnitude = magnitudes
            
        if isinstance(directions, (list, np.ndarray, dict)):
            direction = directions[r_idx]
        else:
            direction = directions
            
        if isinstance(random_mags, (list, np.ndarray, dict)):
            random_mag = random_mag[r_idx]
        else:
            random_mag = random_mags
        
        if magnitude == 0:
            shift = np.zeros(3)
        else:
            if random_mag:
                magnitude = 2 * (np.random.random() - 0.5) * magnitude
                
            if direction is None:
                # generate random 3D vector
                direction = np.random.random(3)
            direction = direction / np.linalg.norm(direction)
            
            # make 3D vector to shift z, y, x coordinates
            shift = magnitude * direction
            
            data += shift
            coords_shifted.loc[select, col_coords] = data
        
        if return_shifts:
            all_shifts[r_idx] = shift
    
    if return_shifts:
        return coords_shifted, all_shifts
    return coords_shifted


def add_false_positives_per_round(coords, proportions, random_props=True, 
                                  col_rounds='rounds', col_coords=['z', 'y', 'x']):
    """
    Add single spots for each round.
    
    Parameters
    ----------
    coords : dataframe
        Spots coordinates with z/y/x and rounds variables.
    proportions : float or list or array
        Proportions of false positive spots added per round.
    random_props : bool or list or array
        If True, random_props are drawn from a uniform distribution between
        0 and the element of `random_props` of each round, else it is this element.
        
    Returns
    -------
    coords_fp : dataframe
        False positives spots coordinates.
    """
        
    sorted_rounds = np.sort(coords[col_rounds].unique())
    rounds = []
    all_fp_coords = []
    for r_idx in sorted_rounds:
        select = coords[col_rounds] == r_idx
        data = coords.loc[select, col_coords].values
        
        if isinstance(proportions, (list, np.ndarray, dict)):
            proportion = proportions[r_idx]
        else:
            proportion = proportions
            
        if isinstance(random_props, (list, np.ndarray, dict)):
            random_prop = random_props[r_idx]
        else:
            random_prop = random_props
        
        if proportion > 1:
            n_fp = proportion
        else:
            if random_prop:
                proportion = np.random.random() * proportion
            n_fp = int(select.sum() * proportion)
        
        fp_coords = []
        for i in range(3):
            mini = data[:, i].min()
            maxi = data[:, i].max()
            diff = maxi - mini
            position = mini + np.random.random(n_fp) * diff
            fp_coords.append(position)
        rounds.extend([r_idx] * n_fp)
        fp_coords = np.vstack(fp_coords).T
        all_fp_coords.append(fp_coords)
    
    all_fp_coords = np.vstack(all_fp_coords)
    fp_coords = pd.DataFrame(data=all_fp_coords, columns=col_coords)
    fp_coords[col_rounds] = rounds
    
    return fp_coords


def update_simulated_coords_from_drift(coords, shifts, codebook, col_rounds='rounds', 
                                       col_spec='species', col_coords=['z', 'y', 'x']):
    """
    Shift coordinates of 'true' molecules given the shifts applied to their
    corresponding simulated spots at different rounds.
    
    Parameters
    ----------
    coords : dataframe
        Spots coordinates with z/y/x, rounds and species.
    shifts : ndarray
        Shifts per round, of shape n_rounds x 3.
    codebook : dict(str)
        Sequences encoding species identites across rounds.
        
    Returns
    -------
    coords_shifted : dataframe
        Spots coordinates that have been shifted
    """
    
    coords_shifted = coords.copy()
    
    for spec in coords[col_spec].unique():
        # get rounds in which the species appears
        spec_code = codebook[spec]
        spec_rounds = [idx for idx, var in enumerate(spec_code) if int(var) == 1]
        # compute the mean shift across rounds
        spec_shifts = np.vstack([shifts[round].reshape((1, -1)) for round in spec_rounds])
        mean_shift = spec_shifts.mean(axis=0)
        # apply mean shift to coordinates
        select = coords_shifted[col_spec] == spec
        coords_shifted.loc[select, col_coords] += mean_shift
    return coords_shifted


def add_false_negatives_per_round(coords, proportions, random_props=True, 
                                  col_rounds='rounds', col_coords=['z', 'y', 'x'],
                                  return_index=True):
    """
    Delete random single spots for each round.
    
    Parameters
    ----------
    coords : dataframe
        Spots coordinates with z/y/x and rounds variables.
    proportions : float or list or array
        Proportions of false negatives (spots deleted), per round.
    random_props : bool or list or array
        If True, random_props are drawn from a uniform distribution between
        0 and the element of `random_props` of each round, else it is this element.
    return_index : bool
        If True, the indices of kept spots are returned.
        
    Returns
    -------
    fn_coords : dataframe
        Initial coordinates with some deleted lines.
    
    Example
    -------
    >>> coords = np.arange(4*4*3).reshape((4*4, 3))
    >>> rounds = np.repeat([0, 1, 2, 3], 4)
    >>> coords = pd.DataFrame(data=coords, columns=['z', 'y', 'x'])
    >>> coords['rounds'] = rounds
    >>> add_false_negatives_per_round(coords, proportions=0.5, 
                                      random_props=False, return_index=False)
    """
        
    sorted_rounds = np.sort(coords[col_rounds].unique())
    rounds = []
    all_fn_coords = []
    for r_idx in sorted_rounds:
        select = coords[col_rounds] == r_idx
        data = coords.loc[select, :]
        
        if isinstance(proportions, (list, np.ndarray, dict)):
            proportion = proportions[r_idx]
        else:
            proportion = proportions
            
        if isinstance(random_props, (list, np.ndarray, dict)):
            random_prop = random_props[r_idx]
        else:
            random_prop = random_props
        
        if proportion > 1:
            n_fn = proportion
        else:
            if random_prop:
                proportion = np.random.random() * proportion
            n_fn = int(select.sum() * proportion)
        
        select = np.full(len(data), True)
        select[np.random.choice(range(len(select)), size=n_fn, replace=False)] = False
        all_fn_coords.append(data.loc[select, :])
    
    if return_index:
        fn_coords = pd.concat(all_fn_coords, axis=0, ignore_index=False)
        index = fn_coords.index.values
        fn_coords.index = np.arange(len(fn_coords))
        return fn_coords, index
    else:
        fn_coords = pd.concat(all_fn_coords, axis=0, ignore_index=True)
        return fn_coords


# ------------ Inspect decoding ------------

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
    source_nodes = []
    target_nodes = []
    if coords_ref is None:
        ind = tree.query_radius(coords, r=r)
        # clean arrays of neighbors from self referencing neighbors
        # and aggregate at the same time
        for i, arr in enumerate(ind):
            neigh = arr[arr != i]
            source_nodes.append([i]*(neigh.size))
            target_nodes.append(neigh)
    else:
        ind = tree.query_radius(coords_ref, r=r)
        # here no need to clean arrays of neighbors from 
        # self referencing neighbors
        for i, neigh in enumerate(ind):
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

    
def make_edges_contributors(
    spots_coords,
    decoded_coords,
    barcodes_selection,
    spots_combinations,
    color_species=None,
):
    """
    Gather data to show localized spots contributing to decoded molecules.

    Parameters
    ----------
    spots_coords : ndarray
        Coordinates of localized spots.
    decoded_coords : ndarray
        Coordinates of decoded spots.
    barcodes_selection : ndarray
        Boolean selector of accapted barcodes.
    spots_combinations : List[List[int]]
        List of lists representing the combinations of spots for each barcode.
    
    Return
    ------
    edges : list[ndarray]
        List of pairs of coordinates.
    edges_colors : list(str)
        Colors of edges.
    
    Example
    -------
    >>> edges, edges_colors = make_edges_contributors(
        decoded_coords,
        optim_results['best_combi'], 
        optim_results['spots_combinations'], 
        colors_decoded)
    """
    
    spots_combis = list(itertools.compress(spots_combinations, barcodes_selection))
    edges = []
    edges_colors = []
    if color_species is None:
        color_species = np.ones(shape=(len(decoded_coords), 4))
    for bcd_id, bcd_coords in enumerate(decoded_coords.values):
        for spot_id in spots_combis[bcd_id]:
            edges.append(np.array([bcd_coords, spots_coords[spot_id]]))
            edges_colors.append(color_species[bcd_id])
    return edges, edges_colors


def make_edges_linkage(
    linkage,
    true_coords,
    decoded_coords,
    col_coords=['z', 'y', 'x'],
):
    """
    Gather data to display linked true and decoded species.

    Parameters
    ----------
    linkage : dict
        Arrays of ids of true positives (`perfect_matches`), incorrect matches ,  
        false negatives (`no_match_true`) and false positives (`no_match_decoded`).
    
    Return
    ------
    edges : list[ndarray]
        List of pairs of coordinates.
    edges_colors : list(str)
        Colors of edges.
        

    Example
    -------
    >>> edges, edges_colors = make_edges_linkage(
        linkage,
        true_coords,
        decoded_coords)
    """
    
    pairs_keys = ['perfect_matches', 'incorrect_matches']
    # TODO: find a nice way to display unmatched spots
    # solo_keys = ['no_match_true', 'no_match_decoded']
    color_names = {
        'perfect_matches': 'green', 
        'incorrect_matches': 'red',
    }
    edges = []
    edges_colors = []
    # need to convert to arrays
    tcoords = true_coords[col_coords].values
    dcoords = decoded_coords[col_coords].values
    for key in pairs_keys:
        pairs_ids = linkage[key]
        linkage_color = color_names[key]
        for src, trg in pairs_ids:
            edges.append(np.array([tcoords[src], dcoords[trg]]))
            edges_colors.append(linkage_color)
    return edges, edges_colors