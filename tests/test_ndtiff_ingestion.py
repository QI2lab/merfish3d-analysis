"""NDTiff ingestion preserves channel order, pixels, and stage metadata."""

import numpy as np
import pandas as pd
import pytest
from ndstorage import NDTiffDataset
from tifffile import imwrite

from merfish3danalysis.cli.qi2lab_microscopes.create_datastore import convert_data
from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.mark.integration
@pytest.mark.parametrize("reversed_channels", [False, True])
def test_ndtiff_ingestion_preserves_channel_pixels_and_stage(
    tmp_path, reversed_channels
):
    root_name = "M1"
    base = f"{root_name}_r0001_tile0000"
    raw_path = tmp_path / f"{base}_1"
    dataset = NDTiffDataset(str(raw_path), summary_metadata={}, writable=True)
    metadata = {
        "Camera-CameraName": "C13440-20CU",
        "Camera-CONVERSION FACTOR COEFF": 1.0,
        "Camera-CONVERSION FACTOR OFFSET": 0.0,
        "PixelSizeUm": 0.2,
        "PixelSizeAffine": "0.2;0;0;0;0.2;0",
        "XPosition_um_Intended": 30.0,
        "YPosition_um_Intended": 20.0,
        "XYStage-TransposeMirrorX": "0",
        "XYStage-TransposeMirrorY": "0",
    }
    images = {}
    for z in range(3):
        for channel, value in [("F-Blue", 100), ("F-Yellow", 200)]:
            pixels = np.full((16, 20), value + z, dtype=np.uint16)
            images[(channel, z)] = pixels
            dataset.put_image(
                {"z": z, "channel": channel},
                pixels,
                {**metadata, "ZPosition_um_Intended": 10.0 + z},
            )
    dataset.finish()
    (raw_path / f"{base}_NDTiffStack.tif").symlink_to("NDTiffStack.tif")
    (tmp_path / "scan_metadata.csv").write_text(
        "root_name,num_r,num_xyz,num_ch,blue_active,yellow_active,red_active,"
        "channels_reversed,z_step_um,yx_pixel_um,binning\n"
        f"{root_name},1,1,2,True,True,False,{reversed_channels},1e0,0.2,1\n"
    )
    pd.DataFrame({"round": [1], "bit": [1]}).to_csv(
        tmp_path / "bit_order.csv", index=False
    )
    pd.DataFrame({"gene_id": ["gene"], "bit01": [1]}).to_csv(
        tmp_path / "codebook.csv", index=False
    )
    imwrite(
        tmp_path / "illuminations.ome.tif",
        np.ones((2, 16, 20), dtype=np.float32),
        metadata={"axes": "CYX"},
    )

    convert_data(
        tmp_path,
        use_illuminations=True,
        channel_names=["fiducial", "readout"],
        noise_map_shape_yx=(16, 20),
    )

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    fiducial, readout = (
        ("F-Yellow", "F-Blue") if reversed_channels else ("F-Blue", "F-Yellow")
    )
    np.testing.assert_array_equal(
        datastore.load_local_corrected_image(tile=0, round=0, return_future=False),
        np.stack([images[(fiducial, z)] for z in range(3)]),
    )
    np.testing.assert_array_equal(
        datastore.load_local_corrected_image(tile=0, bit=0, return_future=False),
        np.stack([images[(readout, z)] for z in range(3)]),
    )
    assert datastore.load_local_image_metadata(tile=0, round=0)["psf_idx"] == 0
    assert datastore.load_local_image_metadata(tile=0, bit=0)["psf_idx"] == 1
    assert len(datastore.channel_psfs) == 2
    stage, camera = datastore.load_local_stage_position_zyx_um(tile=0, round=0)
    np.testing.assert_array_equal(stage, [10, 20, 30])
    np.testing.assert_array_equal(camera, np.eye(4))
    np.testing.assert_allclose(
        datastore.load_local_wavelengths_um(tile=0, round=0), [0.488, 0.520]
    )
    np.testing.assert_allclose(
        datastore.load_local_wavelengths_um(tile=0, bit=0), [0.561, 0.580]
    )
