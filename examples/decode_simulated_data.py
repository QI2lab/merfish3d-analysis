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
from wf_merfish.postprocess import _decode as decode
from wf_merfish.postprocess import _simulation as simulation

dir_main = Path(__file__).parent
dir_images = dir_main / 'images'
dir_images.mkdir(parents=True, exist_ok=True)

codes = [
    '1111000000',
    '1100110000',
    '0000011110',
    '1000000111',
]
species = ['a', 'b', 'c', 'd']
codebook = dict(zip(species, codes))

# ------ simulate coordinates in a cylinder ------
scale = np.array([.115,.115,.115])
n_spots = 5 # number of spots per species
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

simulated_coords.to_csv(dir_main / f'simulated_coords_repeat_id-{repeat_id}.csv', index=False)
true_coords.to_csv(dir_main / f'true_coords_repeat_id-{repeat_id}.csv', index=False)

# # Check simulated data
# viewer = napari.Viewer()
# show_mask = mask + np.random.random(size=mask.shape)
# viewer.add_image(
#     show_mask,
#     name='mask',
#     blending='additive',
#     interpolation2d='nearest',
#     interpolation3d='nearest',
#     scale=scale,
#     contrast_limits=[0, 5],
# )
# viewer.add_points(
#     true_coords[['z', 'y', 'x']],
#     name=f'true coords',
#     size=0.2,
#     blending='additive',
#     face_color='green',
# )
# napari.run()



from scipy.spatial.distance import pdist

radius_coef = 2
maxiter = 800
# z dispersion, x/y dispersion, mean amplitude, std amplitude, sequence error, selection size
weights = np.array([10, 1, 1, 1, 0])
min_spot_sep = 0.5
# min_spot_sep = np.array(localization_params[condi_name]['min_spot_sep'])
dist_params = min_spot_sep * radius_coef
# dist_params = 1
coords = simulated_coords[['z', 'y', 'x']].values
# add some jitter to rescale properly the dispersion
distances = pdist(coords)
min_dist = np.unique(distances)[1]
coords = coords + np.random.uniform(low=-min_dist/4, high=min_dist/4, size=coords.shape)
# fit_vars = np.ones(len(coords)).reshape((-1, 1))
fit_vars = 1000 + np.random.uniform(low=0, high=1, size=len(coords)).reshape((-1, 1))
spot_ids = np.arange(len(coords))
spot_rounds = simulated_coords['rounds'].values.ravel()

optim_results = decode.optimize_spots(
    coords=coords, 
    fit_vars=fit_vars, 
    spot_ids=spot_ids, 
    spot_rounds=spot_rounds, 
    dist_params=dist_params, 
    codebook=codebook,
    weights=weights,
    err_corr_dist=0,
    max_candidates=None,
    history=True, 
    return_extra=True,
    return_contribs=True,
    initialize='maxloss',
    propose_method='iter_bcd',
    n_repeats=1,
    )


# Analyze results, display performance metrics

decoded = optim_results['stats']
# select = np.array([x is not None for x in decoded['species']])
# decoded = decoded.loc[select, :]
select = ~decoded['species'].isna()
decoded = decoded.loc[select, :]
decoded_species = decoded.loc[:, 'species']

select_blanks= np.array([x.startswith('blank') for x in decoded_species])
n_blanks = select_blanks.sum()
print(f'There are {n_blanks} blanks over {select_blanks.size} spots')

col_coords=['z', 'y', 'x']
pairs = simulation.build_rdn(
    coords=decoded[col_coords].values, 
    r=1.5, 
    coords_ref=true_coords[col_coords].values,
    )
linkage = simulation.link_true_decoded_spots(true_coords, decoded, pairs)
stats = simulation.make_linkage_stats(linkage, as_dict=True)


# View results in 3D

decoded_species = decoded['species']
decoded_coords = decoded[['z', 'y', 'x']]
uniq_decoded_species, spec_counts = np.unique(decoded['species'], return_counts=True)

import matplotlib as mpl
cmap = mpl.colormaps['tab20']
colors_dic = {spec: list(cmap(i%cmap.N)) for i, spec in enumerate(decoded_species)}

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
    true_coords[['z', 'y', 'x']], 
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