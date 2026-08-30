from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np
import pytest

from merfish3danalysis.cli.qi2lab_microscopes import fuseall


class FakeArray:
    def __init__(self, shape=(1, 5, 20, 30)) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.chunks = shape
        self.dtype = np.dtype(np.uint16)
        self.metadata = SimpleNamespace(dimension_names=("c", "z", "y", "x"))

    def resize(self, shape) -> None:
        self.shape = shape


@pytest.mark.parametrize("write_ome_tiffs", [False, True])
def test_fuse_all_channels_uses_datastore_and_sequential_mvs_fusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_ome_tiffs: bool,
) -> None:
    fused_root = tmp_path / "qi2labdatastore" / "fused"
    fused_root.mkdir(parents=True)
    datastore = SimpleNamespace(
        tile_ids=["tile0000"],
        round_ids=["round001"],
        bit_ids=["bit002", "bit001"],
        fiducial_folder_name="fiducial",
        voxel_size_zyx_um=(0.4, 0.1, 0.1),
        _fused_root_path=fused_root,
        _image_store_path=fuseall.qi2labDataStore._image_store_path,
        _write_extra_attributes=Mock(),
        load_local_stage_position_zyx_um=Mock(
            return_value=(np.asarray((1.0, 2.0, 3.0)), np.eye(4))
        ),
        load_local_fiducial_image=Mock(return_value=sentinel.fiducial_image),
        load_local_readout_image=Mock(return_value=sentinel.readout_image),
        load_global_coord_xforms_um=Mock(
            return_value=(np.eye(4), np.zeros(3), np.ones(3))
        ),
        load_local_wavelengths_um=Mock(return_value=(0.48, 0.52)),
        load_chromatic_affine_transform_zyx_um=Mock(return_value=np.eye(4)),
    )
    monkeypatch.setattr(fuseall, "qi2labDataStore", Mock(return_value=datastore))
    monkeypatch.setattr(
        fuseall,
        "load_bit_round_transform_zyx_um",
        Mock(return_value=("round002", np.eye(4))),
    )
    monkeypatch.setattr(fuseall.da, "from_array", Mock(return_value=sentinel.lazy))
    store = Mock()
    monkeypatch.setattr(fuseall.da, "store", store)
    monkeypatch.setattr(
        fuseall.si_utils,
        "get_default_spatial_chunksizes",
        Mock(return_value={"z": 256, "y": 256, "x": 256}),
    )
    monkeypatch.setattr(
        fuseall.si_utils,
        "get_sim_from_array",
        Mock(return_value=sentinel.sim),
    )
    monkeypatch.setattr(
        fuseall.msi_utils,
        "get_msim_from_sim",
        Mock(return_value=sentinel.msim),
    )
    set_affine = Mock()
    monkeypatch.setattr(fuseall.msi_utils, "set_affine_transform", set_affine)

    fused_sim = SimpleNamespace(sizes={"z": 5, "y": 20, "x": 30})
    monkeypatch.setattr(
        fuseall.msi_utils,
        "get_sim_from_msim",
        Mock(return_value=fused_sim),
    )
    monkeypatch.setattr(
        fuseall.msi_utils,
        "get_transform_from_msim",
        Mock(return_value=SimpleNamespace(data=np.eye(4)[None])),
    )
    monkeypatch.setattr(
        fuseall.si_utils,
        "get_spatial_dims_from_sim",
        Mock(return_value=("z", "y", "x")),
    )
    monkeypatch.setattr(
        fuseall.si_utils,
        "get_spacing_from_sim",
        Mock(return_value=np.asarray((0.4, 0.1, 0.1))),
    )
    monkeypatch.setattr(
        fuseall.si_utils,
        "get_origin_from_sim",
        Mock(return_value=np.asarray((0.0, 1.0, 2.0))),
    )

    def fuse_channel(**kwargs):
        Path(kwargs["output_zarr_url"]).mkdir(parents=True)
        return sentinel.fused

    fuse = Mock(side_effect=fuse_channel)
    monkeypatch.setattr(fuseall.fusion, "fuse", fuse)
    monkeypatch.setattr(
        fuseall,
        "_direct_zarr_fusion_kwargs",
        Mock(return_value={"backend": "numpy"}),
    )

    multiscales = {"ome": {"multiscales": [{"datasets": [{"path": "0"}]}]}}
    groups = {}

    def open_group(path, mode):
        key = str(path)
        groups.setdefault(key, SimpleNamespace(attrs=dict(multiscales)))
        return groups[key]

    arrays = {}

    def open_array(path, mode):
        key = str(path)
        arrays.setdefault(key, FakeArray())
        return arrays[key]

    monkeypatch.setattr(fuseall.zarr, "open_group", open_group)
    monkeypatch.setattr(fuseall.zarr, "open_array", open_array)

    def copytree(_source, output):
        output.mkdir()

    monkeypatch.setattr(fuseall.shutil, "copytree", copytree)
    export_ome_tiffs = Mock()
    monkeypatch.setattr(fuseall, "export_ome_tiffs", export_ome_tiffs)
    monkeypatch.setattr(fuseall, "tqdm", lambda iterable, **_kwargs: iterable)

    fuseall.fuse_all_channels(tmp_path, write_ome_tiffs=write_ome_tiffs)

    assert [
        Path(call.kwargs["output_zarr_url"]).stem for call in fuse.call_args_list
    ] == ["fiducial.ome", "bit001.ome", "bit002.ome"]
    assert all(
        call.kwargs["transform_key"] == "global_registered"
        for call in fuse.call_args_list
    )
    assert "output_stack_properties" not in fuse.call_args_list[0].kwargs
    assert all(
        "output_stack_properties" in call.kwargs for call in fuse.call_args_list[1:]
    )
    assert set_affine.call_count == 3
    assert store.call_count == 2
    assert (fused_root / "fused_all_channels.ome.zarr").is_dir()
    if write_ome_tiffs:
        export_ome_tiffs.assert_called_once()
    else:
        export_ome_tiffs.assert_not_called()
