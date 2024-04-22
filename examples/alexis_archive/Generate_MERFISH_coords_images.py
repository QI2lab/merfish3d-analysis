"""
Script to simulate a MERFISH experiment, generating random coordinates according
to a codebook, and performing spot localization and decoding.
The details of the simulation are the following:

FISH spots inside a 10 um diameter, 10 um height cylinder (simulating a cell) 
contained in 15 x 20 x 20 um square image volume:

1. Image with zyx sampling as [.115,.115,.115] um  
   blur with light PSF of close to OPM, disregarding tilt 
   (NA_excitation=.01, ex_wvl = .561 um, ri_obj = 1.4, ri_sample=1.4) and  
   (NA_detection = 1.35, em_wvl=.580 um, ri_obj = 1.4, ri_sample=1.4).  
2. Image with zyx experimental sampling of [.7,.108,.108] um, blur with widefield PSF
   with (NA = 1.4, ex_wvl=.561 um, em_wvl = .580 um, ri_obj=1.51, ri_sample=1.4)  
3. Image with zyx experimental sampling of[1.0,.108,.108] um, blur with widefield PSF 
   with (NA = 1.4, ex_wvl=.561 um, em_wvl = .580 um, ri_obj=1.51, ri_sample=1.4)  

These simulation give something close to our instrument (ignoring details of tilt, etc...) 
and the common widefield MERFISH setup with a 60x/NA 1.4 oil objective.

This script is extracted from the script Generate_MERFISH_in_cylinder_vary_density_repeat.py
in the QI2lab repo FISH_analysis_private.
"""

import numpy as np
import pandas as pd
import napari
from pathlib import Path
import tifffile
from localize_psf import camera
from localize_psf.fit_psf import gridded_psf_model, get_psf_coords
import psfmodels as psfm
from tqdm import tqdm
from examples.alexis_archive import _simulation as simulation


visualize_coords = False   # to check simulated coordinates
visualize_images = True   # to check simulated images

# ------ coordinates simulation parameters ------

simu_object = 'cylinder'
# simu_object = 'tri-cylinder'  # 3 stacked cylinders

add_jitter = False  # random independent shift for each spot
add_shift = False   # random shift identical per round
add_fn = False      # delete spots
add_fp = False      # add spots
jitter = 0.05
shift_amp = 0.2 
prop_fn = 0.1
prop_fp = .7

# number of image simulations per condition
# repeat_idxs = np.arange(10)
# or:
repeat_idxs = [0]

# max number of species:
n_species = 121

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

dir_script = Path(__file__).parent
dir_main = dir_script / '../examples/simulated_images' / simu_object
dir_main.mkdir(parents=True, exist_ok=True)
dir_psf = dir_script / '../examples/simulated_images'

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

codebook = dict(zip(species, codes))


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
                    codebook, 
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
                        codebook, 
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
            
            coords_shifted, shifts = simulation.add_drift_per_round(
                simulated_coords, 
                shift_amp, 
                directions=None, 
                random_mags=True,
                col_rounds='bit_idx', 
                col_coords=['z', 'y', 'x'],
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
                col_coords=['z', 'y', 'x'],
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


if visualize_coords:
    viewer = napari.Viewer()

    if simu_object == 'cylinder': 
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
            true_coords[['z', 'y', 'x']],
            name=f'true coords',
            size=0.2,
            blending='additive',
            face_color='green',
        )
    elif simu_object == 'tri-cylinder':
        for i, mask in enumerate(masks):
            show_mask = mask + np.random.random(size=mask.shape)
            viewer.add_image(
                show_mask,
                name=f'mask {i}',
                blending='additive',
                interpolation2d='nearest',
                interpolation3d='nearest',
                scale=scale,
                contrast_limits=[0, 5],
            )
            viewer.add_points(
                true_coords.loc[true_coords['cell_id'] == i, ['z', 'y', 'x']],
                name=f'true coords {i}',
                size=0.2,
                blending='additive',
                face_color='green',
            )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    napari.run()


# ------ Generate images ------

print("Generating images")

for n_spots in all_n_spots:
    for repeat_id in repeat_idxs:
        str_config_spots = f'n_species-{n_species}_n_spots-{n_spots}_repeat_id-{repeat_id}'
        print(str_config_spots)
        simulated_coords = pd.read_csv(dir_coords / f'simulated_coords_{str_config_spots}.csv')
        true_coords = pd.read_csv(dir_coords / f'true_coords_{str_config_spots}.csv')
        
        for condi_name, condi in conditions.items():    
            dz = condi['dz']
            dxy = condi['dxy']
            use_lightsheet = condi['use_lightsheet']
            gt_oversample = condi['gt_oversample']
                
            fname = dir_images / f"{str_config_spots}_widefield_dz={dz:.2f}_lightsheet={use_lightsheet:d}.tiff"
            if fname.exists() and not overwrite_images:
                print(f'Skipping generating existing data for {condi_name}')
            else:
                print(f'Generating data for {condi_name}')
                nz = int(z_fov // dz)
                nxy = int(xy_fov // dxy)

                # generate spots on a finer grid at random locations
                nz_gt = nz * gt_oversample
                nxy_gt = nxy * gt_oversample

                if not use_lightsheet:
                    # widefield PSF
                    nxy_psf = 301
                    path_psf = dir_psf / f'psf_{condi_name}_na-{na}_ni_sample-{ni_sample}_ex_wavelength-{ex_wavelength}_em_wavelength{em_wavelength}_dz-{dz}_dxy-{dxy}_oversample-{gt_oversample}'
                    if not path_psf.exists():
                        print("Generating PSF from vectorial model")
                        coords = get_psf_coords(
                            (nz_gt, nxy_psf, nxy_psf), 
                            (dz / gt_oversample, dxy / gt_oversample, dxy / gt_oversample), 
                            broadcast=True,
                            )
                        psf = gridded_psf_model(em_wavelength, ni_design, model_name="vectorial").model(coords, [1, 0, 0, 0, na, 0])
                        psf = psf / np.sum(psf[nz_gt // 2])
                        print("Saving PSF")
                        tifffile.imwrite(path_psf, psf)
                    else:
                        print("reading PSF")
                        psf = tifffile.imread(path_psf)
                else:
                    path_psf = dir_psf / f'psf_{condi_name}_na-{na}_ni_sample-{ni_sample}_ex_wavelength-{ex_wavelength}_em_wavelength{em_wavelength}_dz-{dz}_dxy-{dxy}_oversample-{gt_oversample}'
                    if not path_psf.exists():
                        print("Generating PSF from vectorial model")
                        nz_psf = int(nz * gt_oversample / 3)
                        nxy_psf = int(nxy_gt / 4)
                        nz_psf = (nz_psf // 2) * 2 + 1 
                        nxy_psf = (nxy_psf // 2) * 2 + 1 
                        # lightsheet PSF
                        oil_lens = {
                            'ni0': ni_design, # immersion medium RI design value
                            'ni': ni_design,  # immersion medium RI experimental value
                            'ns': ni_sample,  # specimen refractive index
                            'tg': 0, # microns, coverslip thickness
                            'tg0': 0 # microns, coverslip thickness design value
                        }
                        ex_lens = {**oil_lens, 'NA': 0.01}
                        em_lens = {**oil_lens, 'NA': na}

                        # the main function
                        ex_psf, em_psf, psf = psfm._core.tot_psf(nx=nxy_psf,
                                                                nz=nz_psf,
                                                                dxy=dxy / gt_oversample,
                                                                dz=dz / gt_oversample,
                                                                pz=0,
                                                                x_offset=0,
                                                                z_offset=0,
                                                                ex_wvl=ex_wavelength,
                                                                em_wvl=em_wavelength,
                                                                ex_params=ex_lens,
                                                                em_params=em_lens,
                                                                psf_func="vectorial")
                        psf = psf / np.sum(psf[nz_psf // 2])
                        
                        print("Saving PSF")
                        tifffile.imwrite(path_psf, psf)
                    else:
                        print("reading PSF")
                        psf = tifffile.imread(path_psf)
                # use less extent for PSF
                psfz, psfy, psfx = psf.shape
                lim = int(psfy / 4)
                psf = psf[:, lim:-lim, lim:-lim]
                    
                imgs = []

                uniq_bits = simulated_coords['bit_idx'].unique()
                uniq_bits.sort()
                for bit_idx in tqdm(uniq_bits):
                    # get current coordinates
                    coords = simulated_coords.loc[simulated_coords['bit_idx'] == bit_idx, ['z', 'y', 'x']].values

                    # half pixel shift
                    coords += 0.5 * scale.reshape((1, -1))
                    coords[:, 0] /= (dz / gt_oversample)
                    coords[:, (1, 2)] /= (dxy / gt_oversample)
                    coords = coords.astype(int)

                    # ground truth image
                    gt = np.zeros((nz_gt, nxy_gt, nxy_gt), dtype=np.float16)
                    for c in [tuple(c) for c in coords]:
                        gt[c] = max_photons

                    # simulate image with PSF blurring + camera noise
                    img, _ = camera.simulated_img(gt, 2, 100, 2, psf, bin_size=gt_oversample)
                    # remove extra z-planes
                    imgs.append(img[::gt_oversample, :, :])
                imgs = np.stack(imgs)

                fname = dir_images / f"{str_config_spots}_widefield_dz={dz:.2f}_lightsheet={use_lightsheet:d}.tiff"
                tifffile.imwrite(fname,
                                 tifffile.transpose_axes(imgs.astype(np.int16), "CZYX", asaxes="TZCYXS"),
                                 imagej=True)


# inspect the true images and the spots coordinates
if visualize_images:
    n_spots = 5
    repeat_id = 0
    condi_name = 'widefield-0.31'
    condi = conditions[condi_name]
    dz = condi['dz']
    dxy = condi['dxy']
    use_lightsheet = condi['use_lightsheet']
    gt_oversample = condi['gt_oversample']
                
    str_config_spots = f'n_species-{n_species}_n_spots-{n_spots}_repeat_id-{repeat_id}'
            
    img = tifffile.imread(dir_images / f"{str_config_spots}_widefield_dz={dz:.2f}_lightsheet={use_lightsheet:d}.tiff")
    img = np.swapaxes(img, 0, 1)
    simulated_coords = pd.read_csv(dir_coords/ f'simulated_coords_{str_config_spots}.csv')
    simulated_coords_ori = pd.read_csv(dir_coords/ f'simulated_coords_original_{str_config_spots}.csv')
    simulated_centers = simulated_coords.loc[:, ['bit_idx', 'z', 'y', 'x']]

    viewer = napari.Viewer()
    view_bits = [4]
    for bit_idx in view_bits:
        coords = simulated_coords.loc[simulated_coords['bit_idx'] == bit_idx, ['z', 'y', 'x']].values
        coords_ori = simulated_coords_ori.loc[simulated_coords_ori['bit_idx'] == bit_idx, ['z', 'y', 'x']].values

        viewer.add_image(
            img[bit_idx], 
            name=f'img bit {bit_idx}', 
            interpolation2d='nearest', 
            interpolation3d='nearest', 
            blending='additive', 
            contrast_limits=[img.min(), img.max()], 
            scale=scale,
            )
        viewer.add_points(
            coords, 
            name=f'true positions bit {bit_idx}', 
            size=scale[-1], 
            face_color='green', 
            edge_color='black', 
            blending='additive',
            )
        viewer.add_points(
            coords_ori, 
            name=f'original positions bit {bit_idx}', 
            size=scale[-1], 
            face_color='red', 
            edge_color='black',
            blending='additive',
            )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    napari.run()
