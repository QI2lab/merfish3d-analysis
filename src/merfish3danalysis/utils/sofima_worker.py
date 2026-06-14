"""Short-lived SOFIMA worker process helpers."""

import os
from typing import Any

import numpy as np


def run_sofima_flow_field_worker(request: dict[str, Any]) -> None:
    """
    Estimate and save SOFIMA flow fields in a short-lived worker process.

    Parameters
    ----------
    request : dict[str, Any]
        Serializable worker request. Required keys are ``datastore_path``,
        ``tile_id``, ``reference_round_id``, ``moving_round_id``, ``gpu_id``,
        ``fixed_zyx``, ``moving_native_zyx``, ``spacing_zyx_um``, and
        ``config``.

    Returns
    -------
    None
        The SOFIMA flow field is written to the datastore.
    """

    gpu_id = int(request["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    mem_fraction = request.get("config", {}).get("xla_memory_fraction")
    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)

    from merfish3danalysis.qi2labDataStore import qi2labDataStore
    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_gpu,
    )
    from merfish3danalysis.utils.sofima_registration import (
        estimate_sofima_flow_field_xyz_px,
    )

    datastore = qi2labDataStore(request["datastore_path"], validate=False)
    tile_id = str(request["tile_id"])
    reference_round_id = str(request["reference_round_id"])
    moving_round_id = str(request["moving_round_id"])
    fixed_zyx = np.asarray(request["fixed_zyx"], dtype=np.float32)
    moving_native_zyx = np.asarray(request["moving_native_zyx"], dtype=np.float32)
    spacing_zyx_um = tuple(float(v) for v in request["spacing_zyx_um"])

    local_transform_zyx_um = datastore.load_local_round_transform_zyx_um(
        tile=tile_id,
        round=moving_round_id,
    )
    if local_transform_zyx_um is None:
        raise RuntimeError(
            f"Missing local transform for tile={tile_id} round={moving_round_id}."
        )

    moving_affine_initialized_zyx = warp_array_to_reference_gpu(
        moving_native_zyx,
        transform_zyx_um=local_transform_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        reference_shape=fixed_zyx.shape,
        gpu_id=0,
    ).astype(np.float32, copy=False)

    sofima_flow_field, metadata = estimate_sofima_flow_field_xyz_px(
        fixed_zyx,
        moving_affine_initialized_zyx,
        config=request.get("config", {}),
    )
    datastore.save_local_sofima_flow_field(
        sofima_flow_field,
        tile=tile_id,
        round=moving_round_id,
        reference_round=reference_round_id,
        map_stride_zyx_px=metadata["map_stride_zyx_px"],
        map_box_start_xyz_px=metadata["map_box_start_xyz_px"],
        map_box_size_xyz_px=metadata["map_box_size_xyz_px"],
        reference_shape_zyx_px=fixed_zyx.shape,
        moving_shape_zyx_px=moving_native_zyx.shape,
        return_future=False,
    )
