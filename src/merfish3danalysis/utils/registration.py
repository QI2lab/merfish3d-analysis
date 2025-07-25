"""
Image registration functions using cucim, scikit-image, and SimpleITK.

This module contains functions for image registration leveraging tools
like cucim, scikit-image, and SimpleITK, optimized for use with
qi2lab 3D MERFISH data.

History:
---------
- **2025/07**: Changed to anisotropic downsampling for registration.
- **2024/12**: Refactored repo structure.
- **2024/07**: Prepared to remove all Dask usage and integrate functions
               into the DataRegistration class as static methods.
- **2024/01**: Updated for qi2lab MERFISH file format v0.1.
- **2023/07**: Initial commit.
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Sequence, Tuple, Optional
import SimpleITK as sitk
#import deeds
import gc
import warpfield

try:
    import cupy as cp # type: ignore
    xp = cp
    CUPY_AVAILABLE = True
except ImportError:
    xp = np
    CUPY_AVAILABLE = False

try:
    from cucim.skimage.registration import phase_cross_correlation # type: ignore
    from cucim.skimage.metrics import structural_similarity # type: ignore
    CUCIM_AVAILABLE = True
except ImportError:
    from skimage.registration import phase_cross_correlation # type: ignore
    from skimage.metrics import structural_similarity # type: ignore
    CUCIM_AVAILABLE = False


# def compute_optical_flow(img_ref: ArrayLike, 
#                          img_trg: ArrayLike) -> ArrayLike:
#     """
#     Compute the optical flow to warp a target image to a reference image.

#     Parameters
#     ----------
#     img_ref: ArrayLike
#         reference image
#     img_trg: ArrayLike
#         moving image

#     Returns
#     -------
#     field: ArrayLike
#         optical flow matrix
#     """

#     field = deeds.registration_fields(
#                 fixed=img_ref, 
#                 moving=img_trg, 
#                 alpha=1.6, 
#                 levels=5, 
#                 verbose=False,
#                 )
#     field = np.array(field)
#     return field

def compute_warpfield(
    img_ref: ArrayLike, 
    img_trg: ArrayLike,
    gpu_id: int = 0
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    
    """
    Compute the warpfield to warp a target image to a reference image.

    Parameters
    ----------
    img_ref: ArrayLike
        reference image
    img_trg: ArrayLike
        moving image
    gpu_id: int, default 0
        GPU ID to use for computation

    Returns
    -------
    warp_field: ArrayLike
        warpfield matrix
    """

    recipe = warpfield.Recipe() # initialized with a translation level, followed by an affine registration level
    recipe.pre_filter.clip_thresh = 1 # clip DC background, if present
    recipe.pre_filter.soft_edge = [4, 32, 32]

    # affine level properties
    recipe.levels[-1].repeats = 0

    recipe.add_level(block_size=[16, 48, 48])
    recipe.levels[-1].block_stride = 0.75
    recipe.levels[-1].smooth.sigmas = [1., 3.0, 3.0]
    recipe.levels[-1].smooth.long_range_ratio = 0.1
    recipe.levels[-1].repeats = 3
    
    recipe.add_level(block_size=[4, 12, 12])
    recipe.levels[-1].block_stride = 0.75
    recipe.levels[-1].smooth.sigmas = [1.5, 5.0, 5.0]
    recipe.levels[-1].smooth.long_range_ratio = 0.1
    recipe.levels[-1].repeats = 3

    warped_image, warp_map, _ = warpfield.register_volumes(
        ref = img_ref, 
        vol = img_trg, 
        recipe = recipe,
        verbose = False,
        gpu_id = gpu_id,
    )
    warped_image = cp.asnumpy(warped_image).astype(np.float32)
    warp_field = cp.asnumpy(warp_map.warp_field).astype(np.float32)
    block_size = cp.asnumpy(warp_map.block_size).astype(np.float32)
    block_stride = cp.asnumpy(warp_map.block_stride).astype(np.float32)
        
    return (warped_image, warp_field, block_size, block_stride)

def apply_transform(
    image1: ArrayLike, 
    image2: ArrayLike,
    transform: sitk.Transform
) -> ArrayLike:
    """
    Apply simpleITK transform

    Parameters
    ----------
    image1: ArrayLike
        reference image
    image2: ArrayLike
        moving image
    transform: sitk.Transform
        simpleITK transform object

    Returns
    -------
    resampled_image: ArrayLike
        transformed moving image
    """
    
    image1_sitk = sitk.GetImageFromArray(image1)
    image2_sitk = sitk.GetImageFromArray(image2)

    # Resample the moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image1_sitk)  # The fixed image is the reference
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)  # Use the transform from the registration

    # Apply the transform to the moving image
    resampled_image = resampler.Execute(image2_sitk)
    
    del image1_sitk, image2_sitk
    gc.collect()

    return sitk.GetArrayFromImage(resampled_image).astype(np.float32)

def compute_rigid_transform(
    image1: ArrayLike, 
    image2: ArrayLike,
    downsample_factors: list[int,int,int] = [2,6,6],
    mask: Optional[ArrayLike] = None,
    projection: Optional[str] = None,
    gpu_id: int = 0
) -> Tuple[sitk.TranslationTransform,Sequence[float]]:
    """
    Calculate initial translation transform using scikit-image
    phase cross correlation. Create simpleITK transform using shift.

    Parameters
    ----------
    image1: ArrayLike
        reference image
    image2: ArrayLike
        moving image
    downsample_factor: list[int,int,int], default = [2,6,6]
        amount of zyx downsampling applied before calling registration
    use_mask: Optional[ArrayLike], default None
        use provided mask 
    projection: Optional[str], default None
        projection method to use

    Returns
    -------
    transform: simpleITK transform
        translation transform
    shift_xyz: Sequence[float]
        xyz shifts in pixels
    """

    with cp.cuda.Device(gpu_id):
   
        if projection is not None:
            if projection == 'z':
                image1 = np.squeeze(np.max(image1,axis=0))
                image2 = np.squeeze(np.max(image2,axis=0))
            elif projection == 'y':
                image1 = np.squeeze(np.max(image1,axis=1))
                image2 = np.squeeze(np.max(image2,axis=1))
                    
        if projection == 'search':
            if CUPY_AVAILABLE and CUCIM_AVAILABLE:
                image1_cp = cp.asarray(image1)
                ref_slice_idx = image1_cp.shape[0]//2
                ref_slice = image1_cp[ref_slice_idx,:,:]
                image2_cp = cp.asarray(image2)
                ssim = []
                for z_idx in range(image1.shape[0]):
                    ssim_slice = structural_similarity(ref_slice.astype(cp.float32),
                                                    image2_cp[z_idx,:].astype(cp.float32),
                                                    data_range=1.0)
                    ssim.append(cp.asnumpy(ssim_slice))
                
                ssim = np.array(ssim)
                print(f"SSIM: {ssim}")
                found_shift = float(ref_slice_idx - np.argmax(ssim))
                print(f"Found shift: {found_shift}")
                del image1_cp, image2_cp, ssim_slice, ssim
            else:
                ref_slice_idx = image1.shape[0]//2
                ref_slice = image1[ref_slice_idx,:,:]
                ssim = []
                for z_idx in range(image1.shape[0]):
                    ssim_slice = structural_similarity(ref_slice.astype(np.float32),
                                                    image2[z_idx,:].astype(np.float32),
                                                    data_range=1.0)
                    ssim.append(ssim_slice)
                
                ssim = np.array(ssim)
                found_shift = float(ref_slice_idx - np.argmax(ssim))

        else:
            # Perform Fourier cross-correlation
            if CUPY_AVAILABLE and CUCIM_AVAILABLE:
                if mask is not None:
                    shift_cp, _, _ = phase_cross_correlation(reference_image=cp.asarray(image1), 
                                                            moving_image=cp.asarray(image2),
                                                            upsample_factor=10,
                                                            reference_mask=mask,
                                                            disambiguate=True)
                else:
                    shift_cp, _, _ = phase_cross_correlation(reference_image=cp.asarray(image1), 
                                                            moving_image=cp.asarray(image2),
                                                            upsample_factor=10,
                                                            disambiguate=True)
                shift = cp.asnumpy(shift_cp)
                del shift_cp
            else:
                if mask is not None:
                    mask = np.zeros_like(image1)
                    shift , _, _ = phase_cross_correlation(reference_image=image1, 
                                                            moving_image=image2,
                                                            upsample_factor=10,
                                                            reference_mask=mask,
                                                            disambiguate=True)
                else:
                    shift , _, _ = phase_cross_correlation(reference_image=image1, 
                                                            moving_image=image2,
                                                            upsample_factor=10,
                                                            disambiguate=True)
        
        # Convert the shift to a list of doubles
        if projection is not None:
            if projection == 'z':
                shift_xyz = [shift[1]*downsample_factors[2],
                            shift[0]*downsample_factors[1],
                            0.]
            elif projection == 'y':
                shift_xyz = [shift_reversed[0],
                            0.,
                            shift_reversed[1]]
            elif projection == 'search':
                shift_xyz = [0.,0.,downsample_factors[0]*found_shift]
        else:
            for i in range(len(shift)):
                if downsample_factors[i] > 1:
                    shift[i] = -1*float(shift[i] * downsample_factors[i])
                else:
                    shift[i] = -1*float(shift[i])
            shift_reversed = shift[::-1]
            shift_xyz = shift_reversed

        # Create an affine transform with the shift from the cross-correlation
        try:
            transform = sitk.TranslationTransform(3, shift_xyz)
        except:
            transform = None
        
        gc.collect()
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


        return transform, shift_xyz

def warp_coordinates(
    coordinates: ArrayLike, 
    tile_translation_transform: sitk.Transform,
    voxel_size_zyx_um: ArrayLike,
    displacement_field_transform: Optional[sitk.Transform] = None
) -> ArrayLike:
    """
    First apply a translation transform to the coordinates, then warp them using a given displacement field.

    Parameters
    ----------
    coordinates: ArrayLike
        List of tuples representing the coordinates.
        MUST be in xyz order!
    voxel_size_zyx_um: ArrayLike
        physical pixel spacing
    displacement_field_transform: Optional[sitk DisplacementField transform], default None
        simpleITK displacement field transform
        
    Returns
    -------
    transformed_coordinates: ArrayLike
        List of tuples representing warped coordinates
        Returned in xyz order!
    """
    voxel_size_xyz_um = voxel_size_zyx_um[::-1]   
    coords_list = [[coord / voxel_size_xyz_um[i] for i, coord in enumerate(point)] for point in coordinates]
    
    
    transformed_coordinates = []
    for coord in coords_list:
        coord_floats = tuple(map(float, coord))
        
        # Apply the translation transform
        translated_physical_coord = tile_translation_transform.TransformPoint(coord_floats)
        
        # Apply the displacement field transform
        if displacement_field_transform is not None:
            warped_coord = displacement_field_transform.TransformPoint(translated_physical_coord)
        
            transformed_coordinates.append(warped_coord)
        else:
            transformed_coordinates.append(translated_physical_coord)
            
    transformed_physical_coords = [[coord * voxel_size_xyz_um[i] for i, coord in enumerate(point)] for point in transformed_coordinates]

    return np.array(transformed_physical_coords)

# def make_flow_vectors(field: Union[ArrayLike,list[ArrayLike]],
#                       mask: ArrayLike = None) -> ArrayLike:
#     """
#     Arrange the results of a optical flow method to display vectors in a 3D volume.
    
#     Parameters
#     ----------
#     field: ArrayLike or list[ArrayLike]
#         Result from scikit-image or cucim ILK or TLV1 methods, or from DEEDS.
#     mask: ArrayLike, default None
#         Boolean mask to select areas where the flow field needs to be computed.
    
#     Returns
#     -------
#     flow_field: ArrayLike
#         A (im_size x 2 x ndim) array indicating origin and final position of voxels.
#     """

#     nz, ny, nx = field[0].shape

#     z_coords, y_coords, x_coords = np.meshgrid(
#         np.arange(nz), 
#         np.arange(ny), 
#         np.arange(nx),
#         indexing='ij',
#         )

#     if mask is not None:
#         origin = np.vstack([z_coords[mask], y_coords[mask], x_coords[mask]]).T
#         shift = np.vstack([field[0][mask], field[1][mask], field[2][mask]]).T 
#     else:
#         origin = np.vstack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()]).T
#         shift = np.vstack([field[0].ravel(), field[1].ravel(), field[2].ravel()]).T 

#     flow_field = np.moveaxis(np.dstack([origin, shift]), 1, 2) 
    
#     return flow_field
