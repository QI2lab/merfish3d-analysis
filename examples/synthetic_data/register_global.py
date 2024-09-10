"""Register polyDT data into global coordinate system

Use multiview-stitcher to globally align and use polyDT data.

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import msi_utils, registration
import dask.diagnostics
import dask.array as da
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import gc
import numpy as np
from tqdm import tqdm


def register_global():
    datastore_path = Path(r"/home/julian/Documents/julian/training_data/readouts/processed")
    datastore = qi2labDataStore(datastore_path=datastore_path)

    # extracted from stage positions
    # tile_sets = [[0, 58], [59, 126], [127, 179], [180, 259], [260, 347], [348, 446]]
    tile_sets = [[2*i,2*i+1] for i in range(7)]

    for tile_set in tile_sets:
        msims = []
        for tile_idx, tile_id in enumerate(
            tqdm(datastore.tile_ids[tile_set[0] : tile_set[1]+1], desc="tile")
        ):
            round_id = datastore.round_ids[0]

            voxel_zyx_um = datastore.voxel_size_zyx_um

            scale = {"z": voxel_zyx_um[0], "y": voxel_zyx_um[1], "x": voxel_zyx_um[2]}

            tile_position_zyx_um = datastore.load_local_stage_position_zyx_um(
                tile_id, round_id
            )

            tile_grid_positions = {
                "z": 0.0,
                "y": tile_position_zyx_um[0],
                "x": tile_position_zyx_um[1],
            }

            im_data = []
            im_data = datastore.load_local_registered_image(
                tile=tile_id, round=round_id, return_future=False
            )

            sim = si_utils.get_sim_from_array(
                da.expand_dims(im_data, axis=0),
                dims=("c", "z", "y", "x"),
                scale=scale,
                translation=tile_grid_positions,
                transform_key="stage_metadata",
            )

            msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
            msims.append(msim)
            del im_data
            gc.collect()

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            with dask.diagnostics.ProgressBar():
                _ = registration.register(
                    msims,
                    reg_channel_index=0,
                    transform_key="stage_metadata",
                    new_transform_key="translation_registered",
                    registration_binning={"z": 4, "y": 4, "x": 4},
                    groupwise_resolution_method="shortest_paths",
                    pre_registration_pruning_method="shortest_paths_overlap_weighted",
                    plot_summary=False,
                )

        for tile_idx, msim in enumerate(msims):
            affine = msi_utils.get_transform_from_msim(
                msim, transform_key="translation_registered"
            ).data.squeeze()
            affine = np.round(affine, 2)
            origin = si_utils.get_origin_from_sim(
                msi_utils.get_sim_from_msim(msim), asarray=True
            )
            spacing = si_utils.get_spacing_from_sim(
                msi_utils.get_sim_from_msim(msim), asarray=True
            )

            datastore.save_global_coord_xforms_um(
                affine_zyx_um=affine,
                origin_zyx_um=origin,
                spacing_zyx_um=spacing,
                tile=tile_idx + tile_set[0],
            )


if __name__ == "__main__":
    register_global()
