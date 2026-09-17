"""Model resolution and transcript matching contracts."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from merfish3danalysis.utils.metrics import calculate_F1_with_radius
from merfish3danalysis.utils.ufish import load_ufish_model, resolve_ufish_weights_path


@pytest.mark.unit
def test_ufish_existing_local_weights_take_precedence_over_alias(tmp_path, monkeypatch):
    weights = tmp_path / "simfish"
    monkeypatch.setattr(Path, "exists", lambda path: path in {Path("simfish"), weights})
    assert resolve_ufish_weights_path("simfish") == Path("simfish")
    model = Mock()
    load_ufish_model(model, weights)
    model.load_weights_from_path.assert_called_once_with(weights)
    model.load_weights.assert_not_called()


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
    monkeypatch, world_mapping
):
    import importlib

    import pandas as pd

    command = importlib.import_module(
        "merfish3danalysis.cli.statphysbio_simulation.calculate_F1"
    )
    # The 3x8x10 image spans Z centers 1, 3, 5 um. Centered ground-truth
    # X,Y columns correspond to image Y,X, with offsets 1.75 and 1.125 um.
    datastore = Mock(voxel_size_zyx_um=[2.0, 0.5, 0.25])
    datastore.load_codebook_parsed.return_value = (["A", "B"], None)
    datastore.load_local_corrected_image.return_value = np.zeros((3, 8, 10))
    datastore.load_global_filtered_decoded_spots.return_value = pd.DataFrame(
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
        datastore.load_global_filtered_decoded_spots.return_value[
            ["global_z", "global_y", "global_x"]
        ] = [[10, 4.25, 5.5], [12, 3.75, 5], [10, 5.25, 7], [14, 3.25, 4]]
    datastore.load_global_coord_xforms_um.return_value = (
        affine,
        origin,
        [2, 0.5, 0.25],
    )
    datastore.load_local_stage_position_zyx_um.return_value = (origin, camera)
    truth = pd.DataFrame(
        {
            "Z": [1.0, 3.0, 5.0, 0.5, 5.5],
            "X": [-0.25, 0.25, 1.25, -0.25, 1.25],
            "Y": [-0.125, 0.375, 0.875, -0.125, 0.875],
            "Gene_label": [1, 2, 1, 1, 1],
        }
    )
    monkeypatch.setattr(command, "qi2labDataStore", Mock(return_value=datastore))
    monkeypatch.setattr(
        command, "resolve_datastore_path", Mock(return_value=Path("/mock/store"))
    )
    monkeypatch.setattr(command.pd, "read_csv", Mock(return_value=truth))
    to_numpy = pd.DataFrame.to_numpy

    def read_only_numpy(frame, *args, **kwargs):
        values = to_numpy(frame, *args, **kwargs)
        values.setflags(write=False)
        return values

    monkeypatch.setattr(pd.DataFrame, "to_numpy", read_only_numpy)
    original_spots = datastore.load_global_filtered_decoded_spots.return_value.copy()
    result = command.calculate_F1(Path("/mock"), search_radius=0.01)
    pd.testing.assert_frame_equal(
        datastore.load_global_filtered_decoded_spots.return_value, original_spots
    )
    # Two exact same-gene matches, two unmatched detections, one missed truth;
    # the two truth points outside the acquired Z support are excluded.
    assert result["True Positives"] == 2
    assert result["False Positives"] == 2
    assert result["False Negatives"] == 1
    assert result["Precision"] == pytest.approx(1 / 2)
    assert result["Recall"] == pytest.approx(2 / 3)
    assert result["F1 Score"] == pytest.approx(4 / 7)
