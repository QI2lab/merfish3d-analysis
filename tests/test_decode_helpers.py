"""Model resolution and transcript matching contracts."""

import numpy as np
import pytest

from merfish3danalysis.utils.metrics import calculate_F1_with_radius
from merfish3danalysis.utils.ufish import resolve_ufish_weights_path


@pytest.mark.unit
def test_ufish_resolves_local_weights_and_leaves_alias_caching_to_upstream(tmp_path):
    weights = tmp_path / "weights.onnx"
    weights.touch()
    assert resolve_ufish_weights_path(weights) == weights
    assert resolve_ufish_weights_path("simfish") == (
        "finetune_models/v1.0.1-simfish_model.onnx"
    )
    assert resolve_ufish_weights_path(None) == resolve_ufish_weights_path("simfish")


@pytest.mark.unit
def test_spatial_matching_is_same_gene_one_to_one_and_closest_first():
    detected = np.array([[0, 0, 0.2], [0, 0, 0.1], [0, 0, 0], [0, 0, 5]])
    truth = np.array([[0, 0, 0], [0, 0, 10]])
    metrics, tp, fp, fn = calculate_F1_with_radius(
        detected, ["A", "A", "B", "A"], truth, ["A", "A"], 0.3
    )
    assert metrics["True Positives"] == 1
    assert metrics["False Positives"] == 3
    assert metrics["False Negatives"] == 1
    assert metrics["F1 Score"] == pytest.approx(1 / 3)
    np.testing.assert_array_equal(tp, detected[[1]])
    np.testing.assert_array_equal(fp, detected[[0, 2, 3]])
    np.testing.assert_array_equal(fn, truth[[1]])


@pytest.mark.unit
def test_empty_spatial_matching_is_a_perfect_match():
    metrics, tp, fp, fn = calculate_F1_with_radius(
        np.empty((0, 3)), [], np.empty((0, 3)), [], 1
    )
    assert metrics["F1 Score"] == 1
    assert tp.shape == fp.shape == fn.shape == (0, 3)


@pytest.mark.unit
@pytest.mark.parametrize("radius, count", [(4.999, 0), (5.0, 1), (5.001, 1)])
def test_matching_radius_is_euclidean_and_inclusive(radius, count):
    # A 3-4-5 triangle has a separation of exactly 5 microns.
    metrics, tp, fp, fn = calculate_F1_with_radius(
        [[0, 3, 4]], ["A"], [[0, 0, 0]], ["A"], radius
    )
    assert metrics["True Positives"] == count
    assert metrics["False Positives"] == 1 - count
    assert metrics["False Negatives"] == 1 - count
    assert metrics["Precision"] == count
    assert metrics["Recall"] == count
    assert metrics["F1 Score"] == count
    assert len(tp) == count
    assert len(fp) == len(fn) == 1 - count


@pytest.mark.unit
@pytest.mark.parametrize("detected_count, truth_count", [(0, 2), (2, 0)])
def test_one_empty_spot_set_has_zero_precision_recall_and_f1(
    detected_count, truth_count
):
    detected = np.arange(detected_count * 3).reshape(detected_count, 3)
    truth = np.arange(truth_count * 3).reshape(truth_count, 3)
    metrics, tp, fp, fn = calculate_F1_with_radius(
        detected, ["A"] * detected_count, truth, ["A"] * truth_count, 1
    )
    assert metrics == {
        "True Positives": 0,
        "False Positives": detected_count,
        "False Negatives": truth_count,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1 Score": 0.0,
    }
    assert tp.shape == (0, 3)
    np.testing.assert_array_equal(fp, detected)
    np.testing.assert_array_equal(fn, truth)


@pytest.mark.integration
@pytest.mark.parametrize("world_mapping", [False, True])
def test_simulation_f1_uses_voxel_centers_axis_order_and_physical_radius(
    image_datastore, tmp_path, world_mapping
):
    import importlib

    import pandas as pd

    command = importlib.import_module(
        "merfish3danalysis.cli.statphysbio_simulation.calculate_F1"
    )
    # The 3x8x10 image spans Z centers 1, 3, 5 um. Centered ground-truth
    # X,Y columns correspond to image Y,X, with offsets 1.75 and 1.125 um.
    datastore = image_datastore
    datastore.voxel_size_zyx_um = [2.0, 0.5, 0.25]
    datastore.codebook = pd.DataFrame(
        {
            "gene_id": ["A", "B"],
            "bit01": [1, 0],
            "bit02": [0, 1],
            "bit03": [1, 0],
            "bit04": [0, 1],
        }
    )
    datastore.save_local_corrected_image(
        np.zeros((3, 8, 10), dtype=np.uint16), tile=0, round=0
    )
    spots = pd.DataFrame(
        {
            "global_z": [0.0, 2.0, 0.0, 4.0],
            "global_y": [1.5, 2.0, 0.0, 3.0],
            "global_x": [1.0, 1.5, 0.0, 2.0],
            "gene_id": ["A", "A", "B", "A"],
        }
    )
    camera = np.eye(4)
    affine = np.eye(4)
    origin = np.zeros(3)
    if world_mapping:
        origin = np.array([7, -2, -1.25])
        camera[1, 1] = -1
        affine = np.array([[1, 0, 0, 3], [0, 0, -1, 4], [0, 1, 0, 5], [0, 0, 0, 1]])
        # Independently calculated world coordinates after stage position,
        # camera Y reflection, then global XY rotation and translation.
        spots[["global_z", "global_y", "global_x"]] = [
            [10, 4.25, 5.5],
            [12, 3.75, 5],
            [10, 5.25, 7],
            [14, 3.25, 4],
        ]
    datastore.save_global_filtered_decoded_spots(spots)
    datastore.save_local_stage_position_zyx_um(origin, camera, tile=0, round=0)
    datastore.save_global_coord_xforms_um(affine, origin, [2, 0.5, 0.25], tile=0)
    truth = pd.DataFrame(
        {
            "Z": [1.0, 3.0, 5.0, 0.5, 5.5],
            "X": [-0.25, 0.25, 1.25, -0.25, 1.25],
            "Y": [-0.125, 0.375, 0.875, -0.125, 0.875],
            "Gene_label": [1, 2, 1, 1, 1],
        }
    )
    truth.to_csv(tmp_path / "GT_spots.csv", index=False)
    acquisition = tmp_path / "sim_acquisition"
    acquisition.mkdir()
    (acquisition / "qi2labdatastore").symlink_to(
        datastore.datastore_path, target_is_directory=True
    )
    result = command.calculate_F1(tmp_path, search_radius=0.01)
    pd.testing.assert_frame_equal(datastore.load_global_filtered_decoded_spots(), spots)
    # Two exact same-gene matches, two unmatched detections, one missed truth;
    # the two truth points outside the acquired Z support are excluded.
    assert result["True Positives"] == 2
    assert result["False Positives"] == 2
    assert result["False Negatives"] == 1
    assert result["Precision"] == pytest.approx(1 / 2)
    assert result["Recall"] == pytest.approx(2 / 3)
    assert result["F1 Score"] == pytest.approx(4 / 7)

    # Explicit ground truth must override the default CSV without changing
    # coordinate conventions.
    alternate_truth = tmp_path / "alternate_ground_truth.csv"
    truth.to_csv(alternate_truth, index=False)
    truth.iloc[0:0].to_csv(tmp_path / "GT_spots.csv", index=False)
    assert (
        command.calculate_F1(tmp_path, search_radius=0.01, gt_path=alternate_truth)
        == result
    )
