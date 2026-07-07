"""
Perform registration on qi2labdatastore. By default creates a max
projection downsampled fiducial OME-TIFF for cellpose parameter optimization.

Shepherd 2025/10 - change to CLI.
Shepherd 2025/07 - rework for multiple GPU support.
Shepherd 2024/11 - rework script to accept parameters.
Shepherd 2024/08 - rework script to use qi2labdatastore object.
"""

from pathlib import Path

import typer

from merfish3danalysis.cli.qi2lab_microscopes._common import qi2lab_datastore_path
from merfish3danalysis.DataRegistration import (
    GlobalFusionConfig,
    GlobalRegistrationConfig,
)
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.sofima_registration import SofimaRegistrationConfig

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def local_register_data(
    root_path: Path,
    num_gpus: int = 1,
    decon: bool = True,
    deformable_registration: bool = True,
    decon_allfiducial: bool = True,
    save_all_fiducial: bool = False,
    overwrite: bool = True,
    crop_yx_decon: int = 2048,
    ufish_model: str | None = None,
    global_registration: bool = True,
    global_registration_only: bool = False,
    create_max_proj_tiff: bool = True,
    registration_diagnostics: bool = False,
    global_registration_binning_zyx: tuple[int, int, int] = (3, 6, 6),
    global_registration_parallel_jobs: int = 1,
    global_registration_scheduler: str = "single-threaded",
    global_registration_affine_round_decimals: int | None = 2,
    global_fusion_n_batch: int = 20,
    global_fusion_n_jobs: int | None = None,
    global_fusion_overlap_in_pixels: int = 64,
    sofima_residual_iterations: int = 2,
    sofima_patch_size_zyx: tuple[int, int, int] = (10, 32, 32),
    sofima_minimum_patch_size_px: int = 4,
    sofima_step_divisor: int = 2,
    sofima_peak_min_distance: int = 2,
    sofima_peak_radius: int = 8,
    sofima_batch_size: int = 32,
    sofima_max_masked: float = 0.75,
    sofima_min_peak_ratio: float = 1.2,
    sofima_min_peak_sharpness: float = 1.2,
    sofima_max_magnitude: float = 30.0,
    sofima_max_deviation: float = 5.0,
    sofima_max_local_z_displacement_px: float = 5.0,
    sofima_subpixel_offsets: tuple[float, float, float] = (-0.5, 0.0, 0.5),
    sofima_subpixel_batch_size: int = 32,
    sofima_normalization_epsilon: float = 1e-6,
    sofima_mesh_dt: float = 0.001,
    sofima_mesh_gamma: float = 0.0,
    sofima_mesh_k0: float = 1.0,
    sofima_mesh_k: float = 0.01,
    sofima_mesh_num_iters: int = 1000,
    sofima_mesh_max_iters: int = 20000,
    sofima_mesh_stop_v_max: float = 0.001,
    sofima_mesh_dt_max: float = 100.0,
    sofima_mesh_start_cap: float = 0.1,
    sofima_mesh_final_cap: float = 10.0,
    verbose: int = 1,
) -> None:
    """Preprocess and register each tile across rounds in local coordinates.

    Parameters
    ----------
    root_path : Path
        Experiment root directory.
    num_gpus : int, default=1
        Number of GPUs available.
    decon : bool, default=True
        Perform readout deconvolution. If False, corrected data are re-saved for
        compatibility instead of deconvolved readouts.
    deformable_registration : bool, default=True
        Perform SOFIMA residual deformable registration.
    decon_allfiducial : bool, default=True
        Perform deconvolution prior to registration for fiducials beyond the first round.
    save_all_fiducial : bool, default=False
        Save all registered fiducial images.
    overwrite : bool, default=True
        Overwrite existing registered data.
    crop_yx_decon : int, default=2048
        Tile size for GPU deconvolution.
    ufish_model : str | None, default=None
        U-FISH model used for feature prediction. If omitted or None, use the
        package default model, simfish. Known aliases include simfish/smfish,
        merfish, seqfish, deepspot, and exseq. A local .onnx/.pth path or
        HuggingFace weights filename may also be used.
    global_registration : bool, default=True
        Perform global tile registration and fused fiducial OME-Zarr creation
        after local preprocessing.
    global_registration_only : bool, default=False
        Skip local preprocessing and rerun only global tile registration and
        fused fiducial OME-Zarr creation on an existing datastore.
    create_max_proj_tiff : bool, default=True
        If True, write the fused fiducial max-projection TIFF when global
        registration runs.
    registration_diagnostics : bool, default=False
        Print detailed registration diagnostics.
    global_registration_binning_zyx : tuple[int, int, int], default=(3, 6, 6)
        Z, Y, X binning passed to multiview-stitcher global registration.
    global_registration_parallel_jobs : int, default=1
        Pairwise registration job count passed to multiview-stitcher.
    global_registration_scheduler : str, default="single-threaded"
        Dask scheduler used around multiview-stitcher global registration.
    global_registration_affine_round_decimals : int | None, default=2
        Decimal precision used when saving global affine transforms. Use None
        to save full precision.
    global_fusion_n_batch : int, default=20
        Number of fusion batches passed to multiview-stitcher.
    global_fusion_n_jobs : int | None, default=None
        Joblib worker count for fusion batches. None derives from the GPU count.
    global_fusion_overlap_in_pixels : int, default=64
        Fusion overlap in pixels.
    sofima_residual_iterations : int, default=2
        Number of SOFIMA residual passes.
    sofima_patch_size_zyx : tuple[int, int, int], default=(10, 32, 32)
        SOFIMA patch size in Z, Y, X pixels.
    sofima_minimum_patch_size_px : int, default=4
        Minimum patch size after clipping to image shape.
    sofima_step_divisor : int, default=2
        Patch-size divisor used to derive SOFIMA flow-grid step.
    sofima_peak_min_distance : int, default=2
        Minimum distance between local cross-correlation peaks.
    sofima_peak_radius : int, default=8
        Peak radius passed to SOFIMA's masked cross-correlation calculator.
    sofima_batch_size : int, default=32
        SOFIMA cross-correlation batch size.
    sofima_max_masked : float, default=0.75
        Maximum masked fraction accepted by SOFIMA cross-correlation.
    sofima_min_peak_ratio : float, default=1.2
        Minimum peak ratio used when cleaning SOFIMA flow vectors.
    sofima_min_peak_sharpness : float, default=1.2
        Minimum peak sharpness used when cleaning SOFIMA flow vectors.
    sofima_max_magnitude : float, default=30.0
        Maximum local flow-vector magnitude in pixels.
    sofima_max_deviation : float, default=5.0
        Maximum local flow-vector deviation in pixels.
    sofima_max_local_z_displacement_px : float, default=5.0
        Maximum residual local Z displacement retained after stabilization.
    sofima_subpixel_offsets : tuple[float, float, float]
        Candidate offsets for subpixel vector refinement.
    sofima_subpixel_batch_size : int, default=32
        Batch size for subpixel vector refinement.
    sofima_normalization_epsilon : float, default=1e-6
        Epsilon used while normalizing local SOFIMA patches.
    sofima_mesh_dt : float, default=0.001
        SOFIMA mesh integration time step.
    sofima_mesh_gamma : float, default=0.0
        SOFIMA mesh damping parameter.
    sofima_mesh_k0 : float, default=1.0
        SOFIMA mesh spring constant for original positions.
    sofima_mesh_k : float, default=0.01
        SOFIMA mesh elastic spring constant.
    sofima_mesh_num_iters : int, default=1000
        Initial SOFIMA mesh relaxation iteration count.
    sofima_mesh_max_iters : int, default=20000
        Maximum SOFIMA mesh relaxation iteration count.
    sofima_mesh_stop_v_max : float, default=0.001
        Velocity threshold used to stop SOFIMA mesh relaxation.
    sofima_mesh_dt_max : float, default=100.0
        Maximum SOFIMA mesh integration time step.
    sofima_mesh_start_cap : float, default=0.1
        Initial SOFIMA mesh displacement cap.
    sofima_mesh_final_cap : float, default=10.0
        Final SOFIMA mesh displacement cap.
    verbose : int, default=1
        Progress verbosity. Set to 0 to suppress routine progress prints.

    """
    from merfish3danalysis.DataRegistration import DataRegistration

    # initialize datastore
    datastore_path = qi2lab_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path)
    print(f"Using datastore at {datastore_path}")

    # initialize registration class
    registration_factory = DataRegistration(
        datastore=datastore,
        decon_fiducial=decon_allfiducial,
        decon_readout=decon,
        perform_deformable_registration=deformable_registration,
        overwrite_registered=overwrite,
        save_all_fiducial_registered=save_all_fiducial,
        num_gpus=num_gpus,
        crop_yx_decon=crop_yx_decon,
        ufish_model=ufish_model,
        global_registration=global_registration,
        registration_diagnostics=registration_diagnostics,
        sofima_config=SofimaRegistrationConfig(
            residual_iterations=sofima_residual_iterations,
            patch_size_zyx=sofima_patch_size_zyx,
            minimum_patch_size_px=sofima_minimum_patch_size_px,
            step_divisor=sofima_step_divisor,
            peak_min_distance=sofima_peak_min_distance,
            peak_radius=sofima_peak_radius,
            batch_size=sofima_batch_size,
            max_masked=sofima_max_masked,
            min_peak_ratio=sofima_min_peak_ratio,
            min_peak_sharpness=sofima_min_peak_sharpness,
            max_magnitude=sofima_max_magnitude,
            max_deviation=sofima_max_deviation,
            max_local_z_displacement_px=sofima_max_local_z_displacement_px,
            subpixel_offsets=sofima_subpixel_offsets,
            subpixel_batch_size=sofima_subpixel_batch_size,
            normalization_epsilon=sofima_normalization_epsilon,
            mesh_dt=sofima_mesh_dt,
            mesh_gamma=sofima_mesh_gamma,
            mesh_k0=sofima_mesh_k0,
            mesh_k=sofima_mesh_k,
            mesh_num_iters=sofima_mesh_num_iters,
            mesh_max_iters=sofima_mesh_max_iters,
            mesh_stop_v_max=sofima_mesh_stop_v_max,
            mesh_dt_max=sofima_mesh_dt_max,
            mesh_start_cap=sofima_mesh_start_cap,
            mesh_final_cap=sofima_mesh_final_cap,
        ),
        global_registration_config=GlobalRegistrationConfig(
            registration_binning_zyx=global_registration_binning_zyx,
            n_parallel_pairwise_regs=global_registration_parallel_jobs,
            dask_scheduler=global_registration_scheduler,
            affine_round_decimals=global_registration_affine_round_decimals,
        ),
        global_fusion_config=GlobalFusionConfig(
            n_batch=global_fusion_n_batch,
            n_jobs=global_fusion_n_jobs,
            overlap_in_pixels=global_fusion_overlap_in_pixels,
        ),
        verbose=verbose,
    )

    if global_registration_only:
        registration_factory.global_register(create_max_proj_tiff=create_max_proj_tiff)
        return

    # run local registration across rounds
    registration_factory.register_all_tiles()

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"LocalRegistered": True})
    datastore.datastore_state = datastore_state


def main() -> None:
    """Run the Typer app."""
    app()


if __name__ == "__main__":
    main()
