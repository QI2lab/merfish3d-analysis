"""
Image registration functions using scikit-image and SimpleITK.

2023/07 - Doug Shepherd
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, List, Sequence, Tuple, Optional
import SimpleITK as sitk
import deeds

CUPY_AVAILABLE = True
try:
    import cupy as cp # type: ignore
    xp = cp
except ImportError:
    xp = np
    CUPY_AVAILABLE = False

CUCIM_AVAILABLE = True
try:
    from cucim.skimage.registration import phase_cross_correlation # type: ignore
    from cucim.skimage.metrics import structural_similarity # type: ignore
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

def compute_rigid_transform(image1: sitk.Image, 
                            image2: sitk.Image,
                            use_mask: Optional[bool] = False,
                            downsample_factor: Optional[float] = 4.0,
                            projection: Optional[str] = None) -> Tuple[sitk.Image,Sequence[float]]:
    """
    Calculate initial translation transform using scikit-image
    phase cross correlation. Create simpleITK transform using shift.

    Parameters
    ----------
    image1: simpleITK image
        reference image
    image2: simpleITK image
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
                mask = cp.zeros_like(image1_np)
                mask[image1_np.shape[0]//2-image1_np.shape[0]//6:image1_np.shape[0]//2+image1_np.shape[0]//6,
                    image1_np.shape[1]//2-image1_np.shape[1]//6:image1_np.shape[1]//2+image1_np.shape[1]//6] = 1
            else:
                mask = None
            
            shift, _, _ = phase_cross_correlation(reference_image=image1_np, 
                                                moving_image=image2_np,
                                                upsample_factor=40,
                                                reference_mask=mask,
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

def warp_coordinates(coordinates: List[Sequence[float]], 
                     translation_transform: sitk.Transform, 
                     displacement_field_transform: Optional[sitk.Transform] = None) -> List[Sequence[float]]:
    """
    First apply a translation transform to the coordinates, then warp them using a given displacement field.

    Parameters
    ----------
    coordinates: List[Sequence[float]] 
        List of tuples representing the coordinates.
        MUST be in xyz order!
    translation_transform: sitk Translation transform
        simpleITK translation transform
    displacement_field_transform: sitk DisplacementField transform
        simpleITK displacement field transform
        
    Returns
    -------
    warped_coordinates: List[Sequence[float]]
        List of tuples representing warped coordinates
        Returned in xyz order!
    """
    
    warped_coordinates = []
    for coord in coordinates:
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

def make_flow_vectors(field: Union[NDArray,List[NDArray]],
                      mask: NDArray = None) -> NDArray:
    """
    Arrange the results of a optical flow method to display vectors in a 3D volume.
    
    Parameters
    ----------
    field: NDArray or List[NDArray]
        Result from scikit-image or cucim ILK or TLV1 methods, or from DEEDS.
    mask: NDArray
        Boolean mask to select areas where the flow field needs to be computed.
    
    Returns
    -------
    flow_field: NDArray
        A (im_size x 2 x ndim) array indicating origin and final position of voxels.
    """

    nz, ny, nx = field[0].shape

    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(nz), 
        np.arange(ny), 
        np.arange(nx),
        indexing='ij',
        )

    if mask is not None:
        origin = np.vstack([z_coords[mask], y_coords[mask], x_coords[mask]]).T
        shift = np.vstack([field[0][mask], field[1][mask], field[2][mask]]).T 
    else:
        origin = np.vstack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()]).T
        shift = np.vstack([field[0].ravel(), field[1].ravel(), field[2].ravel()]).T 

    flow_field = np.moveaxis(np.dstack([origin, shift]), 1, 2) 
    
    return flow_field