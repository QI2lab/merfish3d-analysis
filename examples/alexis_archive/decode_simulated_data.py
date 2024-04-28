import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import napari
from pathlib import Path
import tifffile
from tqdm import tqdm
from localize_psf import camera
from localize_psf.fit_psf import gridded_psf_model, get_psf_coords
import psfmodels as psfm
from examples.alexis_archive import _decode as decode
from examples.alexis_archive import _simulation as simulation

visualize_simulation = True
visualize_decoded = True

simu_objects = ['simple_simu', 'cylinder', 'tri-cylinder']
simu_object = simu_objects[0]

try:
    # case of Python script
    dir_script = Path(__file__).parent
except NameError:
    # case of notebook
    dir_script = Path('./')
dir_main = dir_script / 'simulated_images' / simu_object
dir_main.mkdir(parents=True, exist_ok=True)
dir_psf = dir_script / 'simulated_images'

col_coords = ['z', 'y', 'x']

if simu_object == 'simple_simu':
    codes = [
        '1111000000',
        '1100110000',
        '0000011110',
        '1000000111',
    ]
    species = ['a', 'b', 'c', 'd']
    codebook = dict(zip(species, codes))
    n_species = len(species)

    # ------ simulate coordinates in a cylinder ------
    scale = np.array([.115,.115,.115])
    n_spots = 4 # number of spots per species per image (not per 'cell')
    # FISH spots inside a 10 um diameter, 10 um height cylinder 
    # (simulating a cell) contained in 15 x 20 x 20 um square image
    im_size_um = np.array([15, 20, 20])
    im_size_pix = np.round(im_size_um / scale, decimals=0).astype(int)

    # make mask
    x = np.linspace(-2, 2, im_size_pix[2])
    y = np.linspace(-2, 2, im_size_pix[1])
    xx, yy = np.meshgrid(x, y)
    disk = np.sqrt(xx**2 + yy**2) < 1
    mask = np.full(shape=im_size_pix, fill_value=False)
    z_center = im_size_pix[0] / 2    # z center of the cell
    z_half_thick = 10 / scale[0] / 2 # half the thickness of the cell
    for z_idx in range(int(z_center - z_half_thick), int(z_center + z_half_thick + scale[0])):
        mask[z_idx, :, :] = disk

    repeat_id = 0  # seed
    simulated_coords, true_coords = simulation.generate_barcoded_spots(
        species, 
        n_spots, 
        im_size_pix, 
        codebook, 
        noise=[0, 0], 
        seed=0,
        mask=mask,
        scale=scale,
        )
    simulated_coords.rename(columns={'rounds': 'bit_idx'}, inplace=True)
    true_coords.rename(columns={'rounds': 'bit_idx'}, inplace=True)
    print(simulated_coords)
    
    dir_coords = dir_main / 'coordinates' / f'n_species-{len(species)}_n_spots-{n_spots}'
    dir_coords.mkdir(parents=True, exist_ok=True)
    simulated_coords.to_csv(dir_coords / f'simulated_coords.csv', index=False)
    true_coords.to_csv(dir_coords / f'true_coords.csv', index=False)

else:
    add_jitter = True  # random independent shift for each spot
    add_shift = False   # random shift identical per round
    add_fn = True      # delete spots
    add_fp = True      # add spots
    jitter = 0.1
    shift_amp = 0.1 
    prop_fn = 0.1
    prop_fp = .5

    # number of image simulations per condition
    # repeat_idxs = np.arange(10)
    # or:
    repeat_idxs = [0]

    # max number of species:
    n_species = 130

    # add noise to spot coordinates across rounds (old parameter)
    noise = [0, 0]
    # noise = np.array([2.0, 0.109 * oversampling]) / 2

    # same number of spots across species:
    # 5 * 130 = 650 ~= upper limit of many FISH papers
    # all_n_spots = np.arange(7) + 1
    all_n_spots = [5]

    overwrite_coords = True
    overwrite_images = False

    # ------ image simulation parameters ------

    na = 1.35
    ex_wavelength = 0.561
    em_wavelength = 0.58
    ni_design = 1.51
    ni_sample = 1.4
    max_photons = 300

    conditions = {
        # 'light-sheet':{
        #     'type': 'light-sheet',
        #     'dz': 0.115,
        #     'dxy': 0.115,
        #     'use_lightsheet': True,
        #     'gt_oversample': 5,
        # },
        # uncoment to make images for other microscopy setups:
        'widefield-0.31':{
            'type': 'widefield',
            'dz': 0.31,
            'dxy': 0.088,
            'use_lightsheet': False,
            'gt_oversample': 5,
        },
        # 'widefield-1.0':{
        #     'type': 'widefield',
        #     'dz': 1.0,
        #     'dxy': 0.108,
        #     'use_lightsheet': False,
        #     'gt_oversample': 5,
        # },
    }

    # ------------------------------------------------------------------------
    # --------------------------- Data simulation ----------------------------
    # ------------------------------------------------------------------------

    if not add_jitter:
        jitter = 0
    if not add_shift:
        shift_amp = 0
    if not add_fn:
        prop_fn = 0
    if not add_fp:
        prop_fp = 0

    dir_images = dir_main / 'images' / f'jitter-{jitter}_shift_amp-{shift_amp}_prop_fn-{prop_fn}_prop_fp-{prop_fp}'
    dir_images.mkdir(parents=True, exist_ok=True)
    dir_coords = dir_main / 'coordinates' / f'jitter-{jitter}_shift_amp-{shift_amp}_prop_fn-{prop_fn}_prop_fp-{prop_fp}'
    dir_coords.mkdir(parents=True, exist_ok=True)

    def read_codebook(path_file):
        """
        Read a codebook text file and returns an array of 0 and 1.
        """
        barcodes = []
        with open(path_file) as f:
            for line in f.readlines():
                code = line.strip()
                barcodes.append(code)
        return barcodes


    # ------ Make cell volume ------

    # scale = np.array([.115, .115, .115])
    scale = np.array([.31, 0.088, 0.088])

    if simu_object == 'cylinder':
        # FISH spots inside a 10 um diameter, 10 um height cylinder 
        # (simulating a cell) contained in 15 x 20 x 20 um square image
        im_size_um = np.array([15, 20, 20])
        im_size_pix = np.round(im_size_um / scale, decimals=0).astype(int)
        print('im_size_pix:', im_size_pix)

        # make mask
        x = np.linspace(-2, 2, im_size_pix[2])
        y = np.linspace(-2, 2, im_size_pix[1])
        xx, yy = np.meshgrid(x, y)
        disk = np.sqrt(xx**2 + yy**2) < 1
        mask = np.full(shape=im_size_pix, fill_value=False)
        z_center = im_size_pix[0] / 2    # z center of the cell
        z_half_thick = 10 / scale[0] / 2 # half the thickness of the cell
        for z_idx in range(int(z_center - z_half_thick), int(z_center + z_half_thick + scale[0])):
            mask[z_idx, :, :] = disk
    elif simu_object == 'tri-cylinder':
        # FISH spots inside 3 stacked 10 um diameter, 10 um height cylinders 
        # (simulating 3 cells) contained in 40 x 20 x 20 um square image
        im_size_um = np.array([40, 20, 20])
        im_size_pix = np.round(im_size_um / scale, decimals=0).astype(int)
        print('im_size_pix:', im_size_pix)
        
        # make one mask per cell, in case we want different  
        # gene distributions per cell in the future
        x = np.linspace(-2, 2, im_size_pix[2])
        y = np.linspace(-2, 2, im_size_pix[1])
        xx, yy = np.meshgrid(x, y)
        disk = np.sqrt(xx**2 + yy**2) < 1
        masks = []
        z_half_thick = int(10 / scale[0] / 2) # half the thickness of a cell
        z_centers = (np.array([10, 20, 30]) / scale[0]).astype(int)
        for z_center in z_centers:

            mask = np.full(shape=im_size_pix, fill_value=False)
            # print('low', z_center - z_half_thick, '    high:', z_center + z_half_thick)
            for z_idx in range(z_center - z_half_thick, z_center + z_half_thick + 1):
                # print('    z_idx:', z_idx)
                mask[z_idx, :, :] = disk
            masks.append(np.copy(mask))


    # ------ Make codebook ------
            
    print('absolute path:', Path('./').absolute())

    path_codes = dir_script / '16bitMHD4.txt'
    codes = read_codebook(path_codes)
    species = [f'mRNA {x+1}' for x in range(len(codes))]

    if n_species == -1:
        n_species = len(species)
    else:
        n_species = min(n_species, len(species))
        species = species[:n_species]
        codes = codes[:n_species]

    codebook_generation = dict(zip(species, codes))
    codebook = dict(zip(species, codes))

    # removing the new line characters
    original_cbk = read_codebook(path_codes)
    potential_blanks = set(original_cbk).difference(set(codebook.values()))
    blanks = list(potential_blanks)[-10:]
    blank_cbk = {f'blank_{i}': s for i, s in enumerate(blanks)}
    codebook.update(blank_cbk)


    # ------ Generate coordinates ------

    # full field of view
    if simu_object == 'cylinder':
        z_fov = 15
        xy_fov = 20
        z_cell = 10
        xy_cell = 10
    elif simu_object == 'tri-cylinder':
        z_fov = 40
        xy_fov = 20
        z_cell = 10
        xy_cell = 10

    _cupy_available = True
    try:
        import cupy as cp
    except ImportError:
        cp = np
        _cupy_available = False
    print('_cupy_available:', _cupy_available)

    print("Generating coordinates")
    for n_spots in all_n_spots:
        for repeat_id in repeat_idxs:
            np.random.seed(0)
            str_config_spots = f'n_species-{n_species}_n_spots-{n_spots}_repeat_id-{repeat_id}'
            fname = dir_coords / f'simulated_coords_{str_config_spots}.csv'
            if fname.exists() and not overwrite_coords:
                print(f'Skipping generating already existing data for {str_config_spots}')
            else:  
                print(f'Generate coordinates for {str_config_spots}')
                if simu_object == 'cylinder':
                    simulated_coords, true_coords = simulation.generate_barcoded_spots(
                        species, 
                        n_spots, 
                        im_size_pix, 
                        codebook_generation, 
                        noise=noise, 
                        seed=repeat_id,
                        mask=mask,
                        scale=scale,
                        )
                elif simu_object == 'tri-cylinder':
                    all_simulated_coords = []
                    all_true_coords = []
                    cell_id_simulated = []
                    cell_id_true= []
                    n_masks = len(masks)
                    for i, mask in enumerate(masks):
                        simulated_coords, true_coords = simulation.generate_barcoded_spots(
                            species, 
                            n_spots, 
                            im_size_pix, 
                            codebook_generation, 
                            noise=noise, 
                            seed = repeat_id * n_masks + i,  # change random seed for each cell
                            mask=mask,
                            scale=scale,
                            )
                        all_simulated_coords.append(simulated_coords)
                        all_true_coords.append(true_coords)
                        cell_id_simulated.extend([i] * len(simulated_coords))
                        cell_id_true.extend([i] * len(true_coords))
                    # merge all simulation data
                    simulated_coords = pd.concat(all_simulated_coords, axis=0, ignore_index=True)
                    true_coords = pd.concat(all_true_coords, axis=0, ignore_index=True)
                    simulated_coords['cell_id'] = cell_id_simulated
                    true_coords['cell_id'] = cell_id_true
                    
                # Add drift, FP and FN
                simulated_coords.rename(columns={'rounds': 'bit_idx'}, inplace=True)
                true_coords.rename(columns={'rounds': 'bit_idx'}, inplace=True)
                simulated_coords_ori = simulated_coords.copy()
                simulated_coords_ori.to_csv(dir_coords / f'simulated_coords_original_{str_config_spots}.csv', index=False)
                
                if add_jitter:
                    # add some jitter to rescale properly the dispersion
                    simulated_coords.loc[:, col_coords] = simulated_coords.loc[:, col_coords] + \
                        np.random.uniform(low=-jitter/2, high=jitter/2, size=simulated_coords[col_coords].shape)
                coords_shifted, shifts = simulation.add_drift_per_round(
                    simulated_coords, 
                    shift_amp, 
                    directions=None, 
                    random_mags=True,
                    col_rounds='bit_idx', 
                    col_coords=col_coords,
                    return_shifts=True,
                    )
                coords_shifted, _ = simulation.add_false_negatives_per_round(
                    coords_shifted, 
                    proportions=prop_fn,
                    random_props=False,
                    col_rounds='bit_idx',
                    )
                fp = simulation.add_false_positives_per_round(
                    simulated_coords, 
                    proportions=prop_fp, 
                    random_props=False, 
                    col_rounds='bit_idx', 
                    col_coords=col_coords,
                    )
                simulated_coords = pd.concat([coords_shifted, fp], axis=0, ignore_index=True)
                simulated_coords['bit_idx'] = simulated_coords['bit_idx'].astype(int)
                # quick fix for FP without cell_id, ideally use cell mask ids
                if simu_object == 'tri-cylinder':
                    simulated_coords.loc[~np.isfinite(simulated_coords['cell_id']), 'cell_id'] = 0
                    simulated_coords['cell_id'] = simulated_coords['cell_id'].astype(int)
                true_coords_shifted = simulation.update_simulated_coords_from_drift(true_coords, shifts, codebook)
                
                simulated_coords.to_csv(dir_coords / f'simulated_coords_{str_config_spots}.csv', index=False)
                true_coords.to_csv(dir_coords / f'true_coords_{str_config_spots}.csv', index=False)

if visualize_simulation:
    # Check simulated data
    viewer = napari.Viewer()
    show_mask = mask + np.random.random(size=mask.shape)
    viewer.add_image(
        show_mask,
        name='mask',
        blending='additive',
        interpolation2d='nearest',
        interpolation3d='nearest',
        scale=scale,
        contrast_limits=[0, 5],
    )
    viewer.add_points(
        true_coords[col_coords],
        name=f'true coords',
        size=0.2,
        blending='additive',
        face_color='green',
    )
    viewer.add_points(
        simulated_coords[['bit_idx', 'z', 'y', 'x']],
        name=f'simulated coords',
        size=0.2,
        blending='additive',
        face_color='red',
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    napari.run()


# ------ Decoding -------

radius_coef = 1
# (z dispersion,) x/y dispersion, mean amplitude, std amplitude, sequence error, selection size
# weights = np.array([1, 1, 1, 1, 0])
weights = np.array([5, 1, 1, 1, 0])
min_spot_sep = 0.5
# min_spot_sep = np.array(localization_params[condi_name]['min_spot_sep'])
dist_params = min_spot_sep * radius_coef
# dist_params = 1
coords = simulated_coords[col_coords].values
# add a minimum of jitter to rescale properly the dispersion
if not add_jitter:
    from scipy.spatial.distance import pdist
    distances = pdist(coords)
    min_dist = np.unique(distances)[1]
    print(f'min_dist: {min_dist:.3f}')
    coords = coords + np.random.uniform(low=-min_dist/4, high=min_dist/4, size=coords.shape)
# fit_vars = np.ones(len(coords)).reshape((-1, 1))
fit_vars = 1000 + np.random.uniform(low=0, high=1, size=len(coords)).reshape((-1, 1))
spot_ids = np.arange(len(coords))
spot_rounds = simulated_coords['bit_idx'].values.ravel()

optim_results = decode.optimize_spots(
    coords=coords, 
    fit_vars=fit_vars, 
    spot_ids=spot_ids, 
    spot_rounds=spot_rounds, 
    dist_params=dist_params, 
    codebook=codebook,
    weights=weights,
    err_corr_dist=1,
    max_positive_bits=16,
    max_bcd_per_spot=None,
    history=True, 
    return_extra=True,
    return_contribs=True,
    rescale_used_spots=False,  # no iterations
    trim_network=True,
    # propose_method='iter_bcd',
    propose_method='single_step',
    verbose=1,
    )


# Analyze results, display performance metrics

decoded = optim_results['stats']
n_non_decoded = decoded['species'].isna().sum()
if n_non_decoded != 0:
    print(f'There are {n_non_decoded} molecules')
    select = ~decoded['species'].isna()
    decoded = decoded.loc[select, :]
decoded_species = decoded.loc[:, 'species']

select_blanks= np.array([x.startswith('blank') for x in decoded_species])
n_blanks = select_blanks.sum()
print(f'There are {n_blanks} blanks over {select_blanks.size} spots')

col_coords=['z', 'y', 'x']
pairs = simulation.build_rdn(
    coords=decoded[col_coords].values, 
    r=dist_params, 
    coords_ref=true_coords[col_coords].values,
    )
linkage = simulation.link_true_decoded_spots(true_coords, decoded, pairs)
stats = simulation.make_linkage_stats(linkage, as_dict=True)

print()
print(f'number of species: {n_species}')
print(f'number of spots per species: {n_spots}')
print(f'number of molecules: {len(true_coords)}')
print(f'number of spots: {len(simulated_coords)}')


# View results in 3D

if visualize_decoded:
    decoded_species = decoded['species']
    decoded_coords = decoded[col_coords]
    uniq_decoded_species, spec_counts = np.unique(decoded['species'], return_counts=True)

    import matplotlib as mpl
    cmap = mpl.colormaps['tab20']
    colors_dic = {spec: list(cmap(i%cmap.N)) for i, spec in enumerate(codebook)}

    colors_decoded = np.zeros((len(decoded_coords), 4))
    colors_true= np.zeros((len(true_coords), 4))
    for spec in uniq_decoded_species:
        # colors for decoded spots
        col = colors_dic[spec]
        select = np.array([x == spec for x in decoded_species])
        colors_decoded[select, :] = col
    for spec in np.unique(true_coords['species']):
        # colors for true spots
        col = colors_dic[spec]
        select = true_coords['species'] == spec
        colors_true[select, :] = col

    viewer = napari.Viewer()
    viewer.add_points(
        decoded_coords, 
        name='decoded mRNA',
        size=0.2, 
        face_color=colors_decoded, 
        edge_color='transparent', 
        visible=True,
        scale=(1, 1, 1),
        )

    viewer.add_points(
        true_coords[col_coords], 
        name='true mRNA',
        size=0.2, 
        face_color='transparent', 
        edge_color=colors_true, 
        visible=True,
        scale=(1, 1, 1),
        )

    viewer.add_points(
        coords, 
        name='all coords',
        size=0.05, 
        # face_color=colors_decoded, 
        edge_color='transparent', 
        visible=True,
        scale=(1, 1, 1),
        )

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    napari.run()