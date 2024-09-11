"""Generate deconvolved data and create "fake" local tile registrations.

In this example, we  bypass the standard "DataRegistration" API because 
the Zhuang MOP data is already registered and warped.

For polyDT data, only round 1 is deconvolved. A rigid xyz transform
consisting of all zeros is added to all tiles & rounds for the polyDT data.

For readout data, all tiles and bits are deconvolved plus u-fish predicted.

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils._imageprocessing import chunked_cudadecon
from ufish.api import UFish
import torch
import cupy as cp
from pathlib import Path
import gc
import numpy as np
from tqdm import tqdm
from time import time

def register_local():

    # Create datastore
    datastore_path = Path("/home/julian/Documents/julian/training_data/readouts/processed")
    datastore = qi2labDataStore(datastore_path=datastore_path)
    fiducial_psf = datastore.channel_psfs[0,:]
    # print(fiducial_psf.shape)

    # Deconvolve polyDT data in round 1 across all tiles 
    # We bypass the standard "DataRegistration" API because the Zhuang MOP data is
    # already registered and warped.
    for tile_idx, tile_id in enumerate(tqdm(datastore.tile_ids,desc='tile')):
        round_id = datastore.round_ids[0]
        decon_image = chunked_cudadecon(
            image = datastore.load_local_corrected_image(
                tile=tile_id,
                round=round_id,
                return_future=False
            ),
            psf = fiducial_psf,
            image_voxel_zyx_um=datastore.voxel_size_zyx_um,
            psf_voxel_zyx_um=datastore.voxel_size_zyx_um,
            wavelength_um=datastore.load_local_wavelengths_um(
                tile=tile_id,
                round=round_id
            )[1],
            na=datastore.na,
            ri=datastore.ri
        )
        datastore.save_local_registered_image(
            decon_image,
            tile=tile_id,
            deconvolution=False,
            round=round_id
        )
        
        # Zhuang lab data is pre-registered. Therefore we write zeros to all 
        # rigid transform arrays.
        # We bypass the standard "DataRegistration" API because the Zhuang MOP data is
        # already registered and warped.
        for round_idx, round_id in enumerate(tqdm(datastore.round_ids,desc='round',leave=False)):
            rigid_transform_zyx = [0.,0.,0.]
            datastore.save_local_rigid_xform_xyz_px(
                rigid_xform_xyz_px=np.asarray(rigid_transform_zyx,dtype=np.float32),
                tile=tile_id,
                round=round_id
            )

        # Deconvolve and u-fish predict all tile and bits for readout daqta.
        # We bypass the standard "DataRegistration" API because the Zhuang MOP data is
        # already registered and warped.
        ch_idx = 1
        for bit_idx, bit_id in enumerate(tqdm(datastore.bit_ids,desc='bit')):
            decon_image = chunked_cudadecon(
                image = datastore.load_local_corrected_image(
                    tile=tile_id,
                    bit=bit_id,
                    return_future=False,
                ),
                psf = datastore.channel_psfs[ch_idx,:],
                image_voxel_zyx_um=datastore.voxel_size_zyx_um,
                psf_voxel_zyx_um=datastore.voxel_size_zyx_um,
                wavelength_um=datastore.load_local_wavelengths_um(
                    tile=tile_id,
                    bit=bit_id
                )[1],
                na=datastore.na,
                ri=datastore.ri
            )
            if ch_idx == 1:
                ch_idx = 2
            else:
                ch_idx = 1
            ufish = UFish(device="cuda")
            ufish.load_weights_from_internet()

            ufish_localization, ufish_data = ufish.predict(
                decon_image, axes="zyx", blend_3d=False, batch_size=1
            )

            ufish_localization = ufish_localization.rename(columns={"axis-0": "z"})
            ufish_localization = ufish_localization.rename(columns={"axis-1": "y"})
            ufish_localization = ufish_localization.rename(columns={"axis-2": "x"})

            del ufish
            gc.collect()

            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            roi_z, roi_y, roi_x = 7, 5, 5

            def sum_pixels_in_roi(row, image, roi_dims):
                z, y, x = row["z"], row["y"], row["x"]
                roi_z, roi_y, roi_x = roi_dims
                z_min, y_min, x_min = (
                    max(0, z - roi_z // 2),
                    max(0, y - roi_y // 2),
                    max(0, x - roi_x // 2),
                )
                z_max, y_max, x_max = (
                    min(image.shape[0], z_min + roi_z),
                    min(image.shape[1], y_min + roi_y),
                    min(image.shape[2], x_min + roi_x),
                )
                roi = image[
                    int(z_min) : int(z_max),
                    int(y_min) : int(y_max),
                    int(x_min) : int(x_max),
                ]
                return np.sum(roi)

            ufish_localization["sum_prob_pixels"] = ufish_localization.apply(
                sum_pixels_in_roi,
                axis=1,
                image=ufish_data,
                roi_dims=(roi_z, roi_y, roi_x),
            )
            ufish_localization["sum_decon_pixels"] = ufish_localization.apply(
                sum_pixels_in_roi,
                axis=1,
                image=decon_image,
                roi_dims=(roi_z, roi_y, roi_x),
            )

            ufish_localization["tile_idx"] = tile_idx
            ufish_localization["bit_idx"] = bit_idx + 1
            ufish_localization["tile_z_px"] = ufish_localization["z"]
            ufish_localization["tile_y_px"] = ufish_localization["y"]
            ufish_localization["tile_x_px"] = ufish_localization["x"]

            datastore.save_local_registered_image(
                decon_image,
                tile=tile_id,
                deconvolution=True,
                bit=bit_id
            )
            datastore.save_local_ufish_image(
                ufish_data,
                tile=tile_id,
                bit=bit_id
            )
            datastore.save_local_ufish_spots(
                ufish_localization,
                tile=tile_id,
                bit=bit_id
            )
            
            del (
                decon_image,
                ufish_data,
                ufish_localization,
            )
            gc.collect()
                
if __name__ == "__main__":
    t0 = time()
    register_local()
    t1 = time()
    print(f"Local register time: {t1-t0:.4f}")