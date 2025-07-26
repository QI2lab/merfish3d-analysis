from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.registration import (
    compute_rigid_transform,
    apply_transform
)
from merfish3danalysis.utils.imageprocessing import downsample_image_anisotropic
from merfish3danalysis.utils.rlgc import chunked_rlgc
from pathlib import Path
import warpfield
import numpy as np
import cupy as cp
import SimpleITK as sitk
import gc

def test_reg(root_path: Path):

    with cp.cuda.Device(0):
    
        datastore = qi2labDataStore(root_path / "qi2labdatastore")
        reg_image_decon = datastore.load_local_registered_image(tile=0,round=0,return_future=False)
        mov_image_decon = datastore.load_local_registered_image(tile=0,round=5,return_future=False)

        # reg_image_decon = chunked_rlgc(
        #     image=reg_image,
        #     psf=datastore._psfs[0, :],
        #     gpu_id=0,
        # )

        # mov_image_decon = chunked_rlgc(
        #     image=mov_image,
        #     psf=datastore._psfs[0, :],
        #     gpu_id=0,
        # )

        # datastore.save_local_registered_image(
        #     registered_image = reg_image_decon.clip(0,2**16-1).astype(np.uint16),
        #     tile=0, 
        #     round=0, 
        #     return_future=False
        # )

        # datastore.save_local_registered_image(
        #     registered_image = mov_image_decon.clip(0,2**16-1).astype(np.uint16),
        #     tile=0, 
        #     round=5, 
        #     return_future=False
        # )


        ref_image_decon_float = reg_image_decon.copy().astype(np.float32)
        mov_image_decon_float = mov_image_decon.copy().astype(np.float32)

        downsample_factors = [3,9,9]
        if max(downsample_factors) > 1:
            ref_image_decon_float_ds = downsample_image_anisotropic(
                ref_image_decon_float, downsample_factors
            )
            mov_image_decon_float_ds = downsample_image_anisotropic(
                mov_image_decon_float, downsample_factors
            )
        else:
            ref_image_decon_float_ds = ref_image_decon_float.copy()
            mov_image_decon_float_ds = mov_image_decon_float.copy()


        _, lowres_xyz_shift = compute_rigid_transform(
            ref_image_decon_float_ds,
            mov_image_decon_float_ds,
            downsample_factors=downsample_factors,
            mask = None,
            projection=None
        )
        
        initial_xyz_shift = np.asarray(lowres_xyz_shift)
        print(f"initial xyz shift: {initial_xyz_shift}")

        initial_xyz_transform = sitk.TranslationTransform(3, np.asarray(initial_xyz_shift))
        warped_mov_image_decon_float = apply_transform(
            ref_image_decon_float, mov_image_decon_float, initial_xyz_transform
        )
        del mov_image_decon_float
        gc.collect()


        # translation level properties
        # create a basic recipe:
        recipe = warpfield.Recipe() # initialized with a translation level, followed by an affine registration level
        recipe.pre_filter.clip_thresh = 1 # clip DC background, if present
        recipe.pre_filter.soft_edge = [4, 32, 32]

        # affine level properties
        recipe.levels[-1].repeats = 0

        recipe.add_level(block_size=[16, 48, 48])
        recipe.levels[-1].block_stride = 0.5
        recipe.levels[-1].smooth.sigmas = [1.5, 5.0, 5.0]
        recipe.levels[-1].smooth.long_range_ratio = 0.1
        recipe.levels[-1].repeats = 5
        
        recipe.add_level(block_size=[4, 12, 12])
        recipe.levels[-1].block_stride = 0.75
        recipe.levels[-1].smooth.sigmas = [1.5, 5.0, 5.0]
        recipe.levels[-1].smooth.long_range_ratio = 0.1
        recipe.levels[-1].repeats = 5

        video_path = "output.mp4"
        units_per_voxel = [.315,.098,.098]
        callback = warpfield.utils.mips_callback(units_per_voxel=units_per_voxel)
        moving_reg, warpmap, _ = warpfield.register_volumes(
            ref_image_decon_float, 
            warped_mov_image_decon_float, 
            recipe,
            video_path=video_path, 
            callback=callback
        )

        datastore.save_local_registered_image(
            registered_image = moving_reg.clip(0,2**16-1).astype(np.uint16),
            tile=0, 
            round=6, 
            return_future=False
        )




        

if __name__ == "__main__":
    root_path = Path(r"/mnt/server2/20250702_dual_instrument_WF_MERFISH/")
    test_reg(root_path)
