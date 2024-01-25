"""
Image registration functions using scikit-image and SimpleITK.

2024/01 - Doug Shepherd.
          Updates for qi2lab MERFISH file format v1.0
2023/07 - Doug Shepherd
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, List, Sequence, Tuple, Optional, Dict
import SimpleITK as sitk
import deeds


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

def compute_optical_flow(img_ref: NDArray, 
                         img_trg: NDArray) -> NDArray:
    """
    Compute the optical flow to warp a target image to a reference image.

    Parameters
    ----------
    img_ref: NDArray
        reference image
    img_trg: NDArray
        moving image

    Returns
    -------
    field: NDArray
        optical flow matrix
    """

    field = deeds.registration_fields(
                fixed=img_ref, 
                moving=img_trg, 
                alpha=1.6, 
                levels=5, 
                verbose=False,
                )
    field = np.array(field)
    return field

def apply_transform(image1: sitk.Image, 
                    image2: sitk.Image,
                    transform: sitk.Transform) -> sitk.Image:
    """
    Apply simpleITK transform

    Parameters
    ----------
    image1: sitk.Image
        reference image
    image2: sitk.Image
        moving image
    transform: sitk.Transform
        simpleITK transform object

    Returns
    -------
    resampled_image: sitk.Image
        transformed moving image
    """

    # Resample the moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image1)  # The fixed image is the reference
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)  # Use the transform from the registration

    # Apply the transform to the moving image
    resampled_image = resampler.Execute(image2)

    return resampled_image

def downsample_image(image: sitk.Image, factor: int) -> sitk.Image:
    """
    Isotropic 3D downsample using simpleITK

    Parameters
    ----------
    image: simpleITK image
        image
    factor: int
        isotropic shrink factor

    Returns
    -------
    image_downsampled: simpleITK image
        isotropic downsampled image
    """

    # Downsample the image using BinShrink
    image_downsampled = sitk.BinShrink(image, [factor]*image.GetDimension())

    return image_downsampled

def normalize_histograms(image1: sitk.Image, 
                         image2: sitk.Image) -> sitk.Image:
    """
    Normalize histograms using simpleITK

    Parameters
    ----------
    image1: simpleITK image
        reference image
    image2: simpleITK image
        moving image

    Returns
    -------
    image2_matched: simpleITK image
        moving image histogram matched to reference image
    """

    # Initialize the HistogramMatchingImageFilter
    matcher = sitk.HistogramMatchingImageFilter()

    # Set the number of histogram bins and the number of match points
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)

    # Match the histograms
    image2_matched = matcher.Execute(image2, image1)

    return image2_matched

def compute_rigid_transform(image1: Union[sitk.Image,NDArray], 
                            image2: Union[sitk.Image,NDArray],
                            use_mask: Optional[bool] = False,
                            downsample_factor: Optional[float] = 4.0,
                            projection: Optional[str] = None) -> Tuple[sitk.TranslationTransform,Sequence[float]]:
    """
    Calculate initial translation transform using scikit-image
    phase cross correlation. Create simpleITK transform using shift.

    Parameters
    ----------
    image1: Union[simpleITK image,NDArray]
        reference image
    image2: Union[simpleITK image,NDArray]
        moving image
    use_mask: bool
        use mask for middle 1/3 of image

    Returns
    -------
    transform: simpleITK transform
        translation transform
    shift_xyz: Sequence[float]
        xyz shifts in pixels
    """

    # Convert images to numpy arrays
    image1_np = sitk.GetArrayFromImage(image1)
    image2_np = sitk.GetArrayFromImage(image2)
    
    if projection is not None:
        if projection == 'z':
            image1_np = np.squeeze(np.max(image1_np,axis=0))
            image2_np = np.squeeze(np.max(image2_np,axis=0))
        elif projection == 'y':
            image1_np = np.squeeze(np.max(image1_np,axis=1))
            image2_np = np.squeeze(np.max(image2_np,axis=1))
            
            
    if projection == 'search':
        if CUPY_AVAILABLE and CUCIM_AVAILABLE:
            image1_cp = cp.asarray(image1_np)
            ref_slice_idx = image1_cp.shape[0]//2
            ref_slice = image1_cp[ref_slice_idx,:,:]
            image2_cp = cp.asarray(image2_np)
            ssim = []
            for z_idx in range(image1_np.shape[0]):
                ssim_slice = structural_similarity(ref_slice.astype(cp.uint16),
                                                   image2_cp[z_idx,:].astype(cp.uint16),
                                                   data_range=cp.max(ref_slice)-cp.min(ref_slice))
                ssim.append(cp.asnumpy(ssim_slice))
            
            ssim = np.array(ssim)
            found_shift = float(ref_slice_idx - np.argmax(ssim))
        else:
            ref_slice_idx = image1_np.shape[0]//2
            ref_slice = image1_np[ref_slice_idx,:,:]
            ssim = []
            for z_idx in range(image1_np.shape[0]):
                ssim_slice = structural_similarity(ref_slice.astype(np.uint16),
                                                   image2_np[z_idx,:].astype(np.uint16),
                                                   data_range=np.max(ref_slice)-np.min(ref_slice))
                ssim.append(ssim_slice)
            
            ssim = np.array(ssim)
            found_shift = float(ref_slice_idx - np.argmax(ssim))

    else:
        # Perform Fourier cross-correlation
        if CUPY_AVAILABLE and CUCIM_AVAILABLE:
            if use_mask:
                mask = cp.zeros_like(image1_np)
                mask[:,
                    image1_np.shape[1]//2-image1_np.shape[1]//6:image1_np.shape[1]//2+image1_np.shape[1]//6,
                    image1_np.shape[2]//2-image1_np.shape[2]//6:image1_np.shape[2]//2+image1_np.shape[2]//6] = 1
                shift_cp, _, _ = phase_cross_correlation(reference_image=cp.asarray(image1_np), 
                                                        moving_image=cp.asarray(image2_np),
                                                        upsample_factor=10,
                                                        reference_mask=mask,
                                                        return_error='always',
                                                        disambiguate=True)
            else:
                shift_cp, _, _ = phase_cross_correlation(reference_image=cp.asarray(image1_np), 
                                                        moving_image=cp.asarray(image2_np),
                                                        upsample_factor=10,
                                                        return_error='always',
                                                        disambiguate=True)

            shift = cp.asnumpy(shift_cp)
        else:
            if use_mask:
                mask = np.zeros_like(image1_np)
                mask[image1_np.shape[0]//2-image1_np.shape[0]//6:image1_np.shape[0]//2+image1_np.shape[0]//6,
                    image1_np.shape[1]//2-image1_np.shape[1]//6:image1_np.shape[1]//2+image1_np.shape[1]//6] = 1
                shift , _, _ = phase_cross_correlation(reference_image=image1_np, 
                                                        moving_image=image2_np,
                                                        upsample_factor=10,
                                                        reference_mask=mask,
                                                        return_error='always',
                                                        disambiguate=True)
            else:
                shift , _, _ = phase_cross_correlation(reference_image=image1_np, 
                                                        moving_image=image2_np,
                                                        upsample_factor=10,
                                                        return_error='always',
                                                        disambiguate=True)
    
        # Convert the shift to a list of doubles
        shift = [float(i*-1*downsample_factor) for i in shift]
        shift_reversed = shift[::-1]

    if projection is not None:
        if projection == 'z':
            shift_xyz = [shift_reversed[0],
                         shift_reversed[1],
                         0.]
        elif projection == 'y':
            shift_xyz = [shift_reversed[0],
                         0.,
                         shift_reversed[1]]
        elif projection == 'search':
            shift_xyz = [0.,0.,-downsample_factor*found_shift]
    else:
        shift_xyz = shift_reversed

    # Create an affine transform with the shift from the cross-correlation
    transform = sitk.TranslationTransform(3, shift_xyz)

    return transform, shift_xyz

def warp_localizations(localizations: List[Sequence[float]], 
                       bit_order: Dict,
                       round_ids: List[str],
                       tile_ids: List[str],
                       bit_ids: List[str],
                       rigid_xforms: Dict, 
                       of_xforms: Dict = None) -> List[Sequence[float]]:
    """
    TO DO: Work in progress to apply over all tiles and bits.

    
    First apply a translation transform to the coordinates.
    Second, if available, warp them using a given displacement field.
    
    Parameters
    ----------
    localizations: List[Sequence[float]] 
        List of tuples representing the coordinates.
    bit_order: Dict
        Lookup table that links bits to rounds 
    rigid_xforms: Dict
        Nested dictionary of rigid_xforms
    of_xforms: Dict
        Nested dictionary of of_xforms
        
    Returns
    -------
    localizations: List[Sequence[float]]
        all localizations with transforms applied
    """

    # loop over all tiles and bits
    for tile_id in enumerate(tile_ids):
        for bit_id in enumerate(bit_ids):
            round_id = round_ids[bit_order[tile_id][bit_id]]
            rigid_xform = np.asarray(rigid_xforms[tile_id][round_id])
            translation_transform = sitk.TranslationTransform(3, rigid_xform)

            if of_xforms is not None:
                of_xform = np.asarray(of_xforms[tile_id][round_id])

                of_sitk = sitk.GetImageFromArray(of_xform.transpose(1, 2, 3, 0).astype(np.float64),
                                                            isVector = True)
                final_shape = mov_image_sitk.GetSize()
                optical_flow_sitk = sitk.Resample(of_sitk,final_shape)
                displacement_field_transform = sitk.DisplacementFieldTransform(optical_flow_sitk)

            current_localizations = localizations['detected_coords']['bit_id' == bit_id]

            current_localizations = np.reverse(current_localizations)

            for coord in current_localizations:
                # Convert the coordinate to physical space
                physical_coord = displacement_field_transform.TransformIndexToPhysicalPoint(coord)
                
                # Apply the translation transform
                translated_physical_coord = translation_transform.TransformPoint(physical_coord)
                
                # Apply the displacement field transform
                if displacement_field_transform is not None:
                    warped_coord = displacement_field_transform.TransformPoint(translated_physical_coord)
                

                    warped_coordinates.append(warped_coord)
                else:
                    warped_coordinates.append(translated_physical_coord)

        return warped_coordinates