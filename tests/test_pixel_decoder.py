from math import sqrt

import numpy as np
import pandas as pd
import pytest

from merfish3danalysis.PixelDecoder import PixelDecoder


@pytest.fixture
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("requires CUDA")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires CUDA")
    return cp


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("image_datastore", ["six_bits"], indirect=True)
def test_codebook_single_bit_errors_are_normalized_per_codeword(cuda, image_datastore):
    image_datastore.codebook = pd.DataFrame(
        [["A", 1, 1, 1, 1, 0, 0], ["B", 1, 1, 0, 0, 1, 1], ["C", 0, 0, 1, 1, 1, 1]],
        columns=["gene_id", "bit1", "bit2", "bit3", "bit4", "bit5", "bit6"],
    )
    decoder = PixelDecoder(image_datastore, merfish_bits=6, num_gpus=1, verbose=0)
    # Each original has four on bits. Flipping one bit leaves three or five.
    expected = (decoder._codebook_matrix / 2).tolist()
    for bit in range(6):
        for codeword in decoder._codebook_matrix.tolist():
            codeword[bit] = 1 - codeword[bit]
            expected.append([value / sqrt(sum(codeword)) for value in codeword])

    observed = decoder._normalize_codebook(include_errors=True)

    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(observed, axis=1), 1, atol=1e-7)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("normalization", ["none", "global", "iterative"])
@pytest.mark.parametrize("filter_type", [None, "lp"])
@pytest.mark.parametrize("image_datastore", ["six_bits"], indirect=True)
def test_decode_pixels_matches_known_codewords_magnitudes_and_rejections(
    cuda, image_datastore, normalization, filter_type
):
    image_datastore.codebook = pd.DataFrame(
        [["A", 1, 1, 1, 1, 0, 0], ["B", 1, 1, 0, 0, 1, 1], ["C", 0, 0, 1, 1, 1, 1]],
        columns=["gene_id", "bit1", "bit2", "bit3", "bit4", "bit5", "bit6"],
    )
    decoder = PixelDecoder(
        image_datastore, merfish_bits=6, num_gpus=1, verbose=0, excluded_gene_ids=["C"]
    )
    decoder._filter_type = filter_type
    decoder._pixel_assignment_threshold = 0.5
    a, b, c = decoder._codebook_matrix
    traces = np.stack(
        [
            a * 0.5,
            b * 0.8,
            c,
            np.zeros(6),
            a * 0.49,
            a,
            [1, 0, 0, 0, 0, 0],
            a * 0.75,
            b * 0.5,
            a * 0.8,
            b * 0.75,
            c * 0.8,
        ],
        axis=1,
    ).astype(np.float32)
    background = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
    scale = np.array([2, 4, 8, 16, 32, 64], dtype=np.float32)
    decoder._global_normalization_loaded = normalization != "none"
    decoder._iterative_normalization_loaded = normalization == "iterative"
    decoder._global_background_vector = background
    decoder._global_normalization_vector = scale
    decoder._iterative_background_vector = background
    decoder._iterative_normalization_vector = scale
    if normalization == "iterative":
        # Iterative vectors must take precedence over stale global vectors.
        decoder._global_background_vector = background + 1000
    pixels = (
        traces
        if normalization == "none"
        else background[:, None] + scale[:, None] * traces
    )
    if filter_type == "lp":
        decoder._image_data_lp = pixels.reshape(6, 2, 2, 3)
    else:
        decoder._image_data = pixels.reshape(6, 2, 2, 3)

    decoder._decode_pixels(magnitude_threshold=(1.0, 1.8))

    np.testing.assert_array_equal(
        decoder._decoded_image,
        [[[0, 1, -1], [-1, -1, -1]], [[-1, 0, 1], [0, 1, -1]]],
    )
    np.testing.assert_allclose(
        decoder._magnitude_image.ravel(),
        [1, 1.6, 2, -1, 0.98, 2, 1, 1.5, 1, 1.6, 1.5, 1.6],
        rtol=0,
        atol=0.001,
    )
    np.testing.assert_allclose(
        decoder._distance_image.ravel(),
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        rtol=0,
        atol=0.001,
    )
    np.testing.assert_allclose(
        decoder._scaled_pixel_images.reshape(6, -1), traces, rtol=0, atol=0.001
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("second_gene", "second_tile", "second_xyz", "scores", "expected"),
    [
        ("A", 1, (0.3, 0.4, 0), (0.2, 0.1), [1]),
        ("A", 1, (0.3, 0.4, 0), (0.1, 0.2), [0]),
        ("A", 1, (0.3, 0.4, 0), (0.1, 0.1), [0]),
        ("B", 1, (0.3, 0.4, 0), (0.2, 0.1), [0, 1]),
        ("A", 0, (0.3, 0.4, 0), (0.2, 0.1), [0, 1]),
        ("A", 1, (0.3, 0.4, 0.01), (0.2, 0.1), [0, 1]),
    ],
)
def test_overlap_duplicates_require_same_gene_distinct_tile_and_physical_radius(
    image_datastore, second_gene, second_tile, second_xyz, scores, expected
):
    decoder = PixelDecoder(image_datastore, merfish_bits=4, num_gpus=0, verbose=0)
    # A 3-4-5 triangle reaches the inclusive 0.5-micrometer radius.
    decoder._df_filtered_barcodes = pd.DataFrame(
        {
            "detection": [0, 1],
            "gene_id": ["A", second_gene],
            "tile_idx": [0, second_tile],
            "distance_min": scores,
            "global_x": [0, second_xyz[0]],
            "global_y": [0, second_xyz[1]],
            "global_z": [0, second_xyz[2]],
        }
    )

    decoder._remove_duplicates_in_tile_overlap(radius=0.5)

    assert decoder._df_filtered_barcodes["detection"].tolist() == expected


@pytest.mark.unit
@pytest.mark.parametrize("attribute", ["_df_filtered_barcodes", "_df_barcodes_loaded"])
def test_within_tile_duplicates_follow_physical_clusters_and_preserve_identity(
    image_datastore,
    attribute,
):
    decoder = PixelDecoder(image_datastore, merfish_bits=4, num_gpus=0, verbose=0)
    table = pd.DataFrame(
        [
            # A transitive cluster: endpoints are farther apart than either
            # radius, but each connects through the middle detection.
            (0, "A", 0, 0.0, 0.0, 0.3),
            (1, "A", 0, 0.375, 0.0625, 0.2),
            (2, "A", 0, 0.75, 0.125, 0.1),
            # Same plane: keep both.
            (3, "A", 0, 0.0, 10.0, 0.2),
            (4, "A", 0, 0.0, 10.0625, 0.1),
            # Same coordinates but different genes or tiles: keep all.
            (5, "A", 0, 0.0, 20.0, 0.2),
            (6, "B", 0, 0.375, 20.0, 0.1),
            (7, "A", 1, 0.375, 20.0, 0.1),
            # Inclusive XY and Z bounds, with a score tie: first row wins.
            (8, "A", 0, 0.0, 30.0, 0.2),
            (9, "A", 0, 0.5, 30.125, 0.2),
            # Beyond the Z or XY bound: keep all three.
            (10, "A", 0, 0.0, 40.0, 0.3),
            (11, "A", 0, 0.501, 40.0, 0.1),
            (12, "A", 0, 0.25, 40.126, 0.1),
        ],
        columns=[
            "detection",
            "gene_id",
            "tile_idx",
            "global_z",
            "global_x",
            "distance_min",
        ],
    )
    table["global_y"] = 7.0
    setattr(decoder, attribute, table)

    decoder._remove_duplicates_within_tile(radius_xy=0.125, radius_z=0.5)

    assert getattr(decoder, attribute)["detection"].tolist() == [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
        12,
    ]


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("is_3d", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("image_datastore", ["six_bits"], indirect=True)
def test_decode_and_extract_generated_transcripts_with_physical_coordinates(
    cuda, image_datastore, is_3d, empty
):
    image_datastore.codebook = pd.DataFrame(
        [["A", 1, 1, 1, 1, 0, 0], ["B", 1, 1, 0, 0, 1, 1]],
        columns=["gene_id", "bit1", "bit2", "bit3", "bit4", "bit5", "bit6"],
    )
    decoder = PixelDecoder(
        image_datastore,
        merfish_bits=6,
        num_gpus=1,
        verbose=0,
        decode_mode="3d" if is_3d else "2d",
        z_range=[2, 5],
    )
    decoder._image_data = np.zeros((6, 3, 8, 10), dtype=np.float32)
    if not empty:
        decoder._image_data[:, :2, 1:3, 2:4] = (
            decoder._codebook_matrix[0, :, None, None, None] * 0.75
        )
        decoder._image_data[:, 2:, 5:7, 6:8] = (
            decoder._codebook_matrix[1, :, None, None, None] * 0.5
        )
    decoder._filter_type = None
    decoder._pixel_assignment_threshold = 0.5
    decoder._transcript_distance_threshold = 0.5
    decoder._tile_idx = 0
    decoder._spacing = np.array([0.5, 0.2, 0.4])
    decoder._origin = np.array([10, 20, 30])
    decoder._camera_to_stage_affine = np.diag([1, -1, 1, 1])
    decoder._affine = np.eye(4)
    decoder._affine[:3, 3] = [1, 2, -3]

    decoder._decode_pixels(magnitude_threshold=(1, 2))
    decoder._extract_barcodes(minimum_pixels=4, maximum_pixels=8)

    observed = decoder.decoded_barcodes.sort_values(["gene_id", "z"])
    if empty:
        assert observed.empty
        return
    assert observed["gene_id"].tolist() == (["A", "B"] if is_3d else ["A", "A", "B"])
    np.testing.assert_array_equal(observed["area"], [8, 4] if is_3d else [4, 4, 4])
    np.testing.assert_array_equal(
        observed[["z", "y", "x"]],
        [[2.5, 1.5, 2.5], [4, 5.5, 6.5]]
        if is_3d
        else [[2, 1.5, 2.5], [3, 1.5, 2.5], [4, 5.5, 6.5]],
    )
    # Centroids include the two cropped source planes before applying spacing.
    expected = (
        [[12.25, -18.3, 28], [13, -19.1, 29.6]]
        if is_3d
        else [[12, -18.3, 28], [12.5, -18.3, 28], [13, -19.1, 29.6]]
    )
    np.testing.assert_allclose(
        observed[["global_z", "global_y", "global_x"]], expected, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        observed["signal_mean"], [0.75, 0.5] if is_3d else [0.75, 0.75, 0.5]
    )
    np.testing.assert_array_equal(observed["bkd_mean"], 0)
    np.testing.assert_allclose(observed["distance_min"], 0, atol=1e-7)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("stored_global", [False, True])
def test_load_bit_data_weights_then_warps_then_crops_generated_readouts(
    cuda, image_datastore, stored_global
):
    decoder = PixelDecoder(
        image_datastore,
        merfish_bits=4,
        num_gpus=1,
        verbose=0,
        decode_mode="3d",
        z_range=[1, 4],
    )
    decoder._tile_idx = 0
    decoder._decode_run_key = "current"
    native = np.arange(6 * 8 * 12, dtype=np.float32).reshape(6, 8, 12)
    prediction = np.broadcast_to(
        np.arange(1, 9, dtype=np.float32)[None, :, None] / 8, native.shape
    )
    datastore = image_datastore
    datastore.voxel_size_zyx_um = [0.5, 0.25, 0.25]
    for bit, image in enumerate((native, native * 2, native, native * 2)):
        datastore.save_local_corrected_image(image, tile=0, bit=bit)
        datastore.save_local_feature_predictor_image(prediction, tile=0, bit=bit)
        datastore.save_local_round_linker(2, tile=0, bit=bit)
    round_transform = np.eye(4)
    round_transform[:3, 3] = [0.5, 0.25, 0.5]
    datastore.save_local_round_transform_zyx_um(round_transform, tile=0, round=1)
    chromatic = np.diag([1, -1, 1, 1]).astype(float)
    chromatic[:3, 3] = [0, 2, -0.25]
    datastore.save_chromatic_affine_transforms_zyx_um(
        {
            "estimator": "decoded_rna_on_bit_weighted_centroid_yx_affine_graph",
            "decode_run_key": "previous",
            "channels": {"red": {"wavelength_um": 0.67, "affine_zyx_um": chromatic}},
        }
    )
    datastore.save_local_stage_position_zyx_um(
        [10, 20, 30], np.diag([1, -1, 1, 1]), tile=0, round=0
    )
    global_affine = np.eye(4)
    global_affine[:3, 3] = [1, 2, 3]
    if stored_global:
        datastore.save_global_coord_xforms_um(
            global_affine, [10, 20, 30], [0.5, 0.25, 0.25], tile=0
        )
    decoder._load_bit_data()

    # Inverse chromatic after round pull: native Z=z+1, Y=7-y, X=x+3.
    # Crop output Z=1:4 only after warping, hence native Z=2:5.
    expected = np.zeros((3, 8, 12), dtype=np.float32)
    expected[:, :, :9] = (native * prediction)[2:5, ::-1, 3:]
    np.testing.assert_allclose(decoder._image_data[0], expected, rtol=0, atol=1e-5)
    np.testing.assert_allclose(decoder._image_data[1], expected * 2, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(decoder._origin, [10, 20, 30])
    np.testing.assert_array_equal(decoder._spacing, [0.5, 0.25, 0.25])
    np.testing.assert_array_equal(native.ravel(), np.arange(native.size))
    landmark = decoder._warp_pixel(
        np.zeros(3),
        decoder._spacing,
        decoder._origin,
        decoder._affine,
        decoder._camera_to_stage_affine,
    )
    np.testing.assert_array_equal(
        landmark, [11, -18, 33] if stored_global else [10, -20, 30]
    )


@pytest.mark.unit
@pytest.mark.parametrize("image_datastore", ["six_bits"], indirect=True)
def test_codebook_thresholds_follow_four_on_bit_geometry(image_datastore):
    image_datastore.codebook = pd.DataFrame(
        [
            ["A", 1, 1, 1, 1, 0, 0],
            ["Blank1", 1, 1, 0, 0, 1, 1],
            ["single", 1, 0, 0, 0, 0, 0],
        ],
        columns=["gene_id", "bit1", "bit2", "bit3", "bit4", "bit5", "bit6"],
    )
    genes, matrix = image_datastore.load_codebook_parsed()
    assert genes == ["A", "Blank1", "single"]
    np.testing.assert_array_equal(
        matrix, [[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0]]
    )
    decoder = PixelDecoder(image_datastore, merfish_bits=6, num_gpus=0, verbose=0)

    assert decoder._gene_ids == ["A", "Blank1"]
    assert decoder._blank_count == 1
    # Unit-vector distance squared is 2 - 2*cos(theta). Two lost on bits
    # give cos(theta)=1/sqrt(2); two gained bits give sqrt(2/3).
    assert decoder._pixel_assignment_threshold == pytest.approx(sqrt(2 - sqrt(2)))
    assert decoder._transcript_distance_threshold == pytest.approx(
        sqrt(2 - 2 * sqrt(2 / 3))
    )
    np.testing.assert_array_equal(
        decoder._codebook_matrix, [[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "threshold, expected", [(-1, 2 / 3), (0.5, 0.5), (0.9, np.inf)]
)
def test_lr_fdr_uses_retained_counts_and_codebook_blank_fraction(threshold, expected):
    spots = pd.DataFrame(
        {
            "gene_id": ["A", "B", "A", "Blank1", "bLaNk2"],
            "predicted_probability": [0.9, 0.8, 0.5, 0.7, 0.5],
        }
    )
    # Two coding and two blank codewords. At threshold 0.5 (strict >),
    # retained counts are two coding and one blank: (1/2)/(2/2) = 1/2.
    observed = PixelDecoder._calculate_lr_fdr(
        spots, threshold=threshold, blank_count=2, barcode_count=4
    )
    assert observed == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize(
    "target, expected_ids", [(0.1, [0, 1]), (0.5, [0, 1, 2, 3]), (2.0, list(range(5)))]
)
def test_blank_fraction_histogram_includes_upper_edges_and_matches_known_counts(
    cuda, image_datastore, target, expected_ids
):
    image_datastore.codebook = pd.DataFrame(
        [["A", 1, 1, 1, 1], ["Blank1", 1, 1, 1, 1]],
        columns=["gene_id", "bit1", "bit2", "bit3", "bit4"],
    )
    decoder = PixelDecoder(image_datastore, merfish_bits=4, num_gpus=1, verbose=0)
    decoder._df_barcodes_loaded = pd.DataFrame(
        {
            "detection": range(5),
            "gene_id": ["A", "A", "A", "Blank1", "Blank1"],
            "magnitude_mean": [0.5, 0.5, 1.5, 1.5, 2.0],
            "area": [4, 4, 4, 4, 4],
            "distance_min": [0.1] * 5,
        }
    )

    decoder._filter_all_barcodes_blank_fraction(
        target_gross_misid_rate=target,
        intensity_bins=[0, 1, 1.75, 2],
        voxel_number_bins=[0, 4],
        vector_distance_bins=[0, 0.1],
    )

    result = decoder._blank_fraction_filter_results
    np.testing.assert_array_equal(result["all_histogram"].ravel(), [2, 2, 1])
    np.testing.assert_array_equal(result["blank_histogram"].ravel(), [0, 1, 1])
    np.testing.assert_array_equal(
        result["blank_fraction_histogram"].ravel(), [0, 0.5, 1]
    )
    # Gross rate uses blank/codebook counts and all retained transcripts.
    np.testing.assert_allclose(
        result["threshold_sweep"]["gross_misid_rate"], [0, 0.5, 0.8]
    )
    assert decoder._df_filtered_barcodes["detection"].tolist() == expected_ids


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("image_datastore", ["six_bits"], indirect=True)
def test_iterative_normalization_saves_and_keeps_memory_consistent(
    cuda, image_datastore, empty
):
    image_datastore.codebook = pd.DataFrame(
        {
            "gene_id": ["A", "B"],
            **{
                f"bit{index + 1:02}": values
                for index, values in enumerate(
                    [[1, 0], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1]]
                )
            },
        }
    )
    decoder = PixelDecoder(
        image_datastore,
        merfish_bits=6,
        num_gpus=1,
        verbose=0,
        decode_mode="3d",
        normalization_features="all",
    )
    decoder._iterative_background_vector = None
    decoder._iterative_normalization_vector = None
    decoder._global_background_vector = cuda.asarray(
        [1, 2, 3, 4, 5, 6], dtype=cuda.float32
    )
    decoder._global_normalization_vector = cuda.asarray(
        [10, 20, 30, 40, 50, 60], dtype=cuda.float32
    )
    decoder._iterative_normalization_loaded = False
    decoder._df_barcodes_loaded = pd.DataFrame(
        [
            ["A", 1, 2, 3, 4, 20, 30, 40, 50, 5, 6],
            ["B", 3, 4, 5, 6, 1, 2, 60, 70, 80, 90],
        ],
        columns=["gene_id", "on_bit_1", "on_bit_2", "on_bit_3", "on_bit_4"]
        + [f"bit{i:02d}_mean_intensity" for i in range(1, 7)],
    )
    if empty:
        decoder._df_barcodes_loaded = decoder._df_barcodes_loaded.iloc[:0]

    decoder._iterative_normalization_vectors()

    normalization, background = image_datastore.load_decode_normalization_vectors(
        None, "iterative"
    )
    # Per-bit on/off medians, with no off-bit samples for bits 3 and 4.
    expected_norm = [10, 20, 30, 40, 50, 60] if empty else [20, 30, 50, 60, 80, 90]
    expected_background = [1, 2, 3, 4, 5, 6] if empty else [1, 2, 0, 0, 5, 6]
    np.testing.assert_array_equal(normalization, expected_norm)
    np.testing.assert_array_equal(background, expected_background)
    np.testing.assert_array_equal(
        cuda.asnumpy(decoder._iterative_normalization_vector), expected_norm
    )
    np.testing.assert_array_equal(
        cuda.asnumpy(decoder._iterative_background_vector), expected_background
    )
    assert decoder._iterative_normalization_loaded


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("use_flow", [False, True])
@pytest.mark.parametrize("use_chromatic", [False, True])
@pytest.mark.parametrize("image_datastore", ["six_bits"], indirect=True)
def test_bits_decode_in_round_one_before_world_mapping(
    cuda, image_datastore, use_flow, use_chromatic
):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    datastore.voxel_size_zyx_um = [0.5, 0.25, 0.25]
    codebook = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]], dtype=np.float32)
    datastore.codebook = pd.DataFrame(
        {"gene_id": ["A", "B"], **{f"bit{i + 1:02}": codebook[:, i] for i in range(6)}}
    )
    shape = (8, 12, 32)
    reference = np.zeros((6, *shape), dtype=np.float32)
    reference[:, 2:4, 3:5, 12:14] = codebook[0, :, None, None, None] * 0.75
    reference[:, 4:6, 7:9, 20:22] = codebook[1, :, None, None, None] * 0.75
    # A row-dependent X displacement is a non-rigid local refinement.
    flow = np.zeros((3, *shape), dtype=np.float32)
    flow[0, :, 6:, :] = 1
    round_transform = np.eye(4)
    round_transform[:3, 3] = [0.5, 0.25, -0.5]
    chromatic = np.eye(4)
    chromatic[2, 1] = 1  # forward wavelength correction: X <- X + Y
    chromatic[2, 3] = 0.25
    for bit in range(6):
        native = np.zeros(shape, dtype=np.float32)
        if bit == 0:
            native[:] = reference[bit]
        else:
            # Derive native sample locations directly in pixels. The local
            # translation is (+1,+1,-2). The inverse chromatic shear acts on
            # translated Y, after the row-dependent flow displacement.
            for z, y, x in np.argwhere(reference[bit] > 0):
                native_x = x + int(use_flow and y >= 6) - 2
                if use_chromatic and bit >= 3:
                    native_x -= y + 2
                native[z + 1, y + 1, native_x] = reference[bit, z, y, x]
        datastore.save_local_round_linker(1 if bit == 0 else 2, tile=0, bit=bit)
        datastore.save_local_wavelengths_um(
            (0.5, 0.6) if bit < 3 else (0.6, 0.7), tile=0, bit=bit
        )
        datastore.save_local_corrected_image(native, tile=0, bit=bit)
        datastore.save_local_feature_predictor_image(
            np.ones(shape, dtype=np.float32), tile=0, bit=bit
        )
    datastore.save_local_round_transform_zyx_um(round_transform, tile=0, round=1)
    if use_chromatic:
        datastore.save_chromatic_affine_transforms_zyx_um(
            {
                "estimator": "decoded_rna_on_bit_weighted_centroid_yx_affine_graph",
                "reference_wavelength_um": 0.6,
                "channels": {
                    "reference": {"wavelength_um": 0.6, "affine_zyx_um": np.eye(4)},
                    "readout": {"wavelength_um": 0.7, "affine_zyx_um": chromatic},
                },
            }
        )
    if use_flow:
        datastore.save_local_sofima_flow_field(
            flow,
            tile=0,
            round=1,
            reference_round=0,
            map_stride_zyx_px=(1, 1, 1),
            map_box_start_xyz_px=(0, 0, 0),
            map_box_size_xyz_px=shape[::-1],
            reference_shape_zyx_px=shape,
            moving_shape_zyx_px=shape,
            sofima_status="ok",
        )
    datastore.save_local_stage_position_zyx_um(
        [10, 20, 30], np.diag([1, -1, 1, 1]), tile=0, round=0
    )
    global_affine = np.array([[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]])
    datastore.save_global_coord_xforms_um(
        global_affine, [10, 20, 30], [0.5, 0.25, 0.25], tile=0
    )
    datastore = qi2labDataStore(datastore.datastore_path, validate=False)
    decoder = PixelDecoder(
        datastore, merfish_bits=6, num_gpus=1, verbose=0, decode_mode="3d"
    )
    decoder._tile_idx = 0
    decoder._filter_type = None
    decoder._pixel_assignment_threshold = 0.5
    decoder._transcript_distance_threshold = 0.5

    decoder._load_bit_data()
    np.testing.assert_array_equal(decoder._image_data, reference)
    decoder._decode_pixels(magnitude_threshold=(1, 2))
    expected_labels = np.full(shape, -1, dtype=np.int16)
    expected_labels[2:4, 3:5, 12:14] = 0
    expected_labels[4:6, 7:9, 20:22] = 1
    np.testing.assert_array_equal(decoder._decoded_image, expected_labels)
    decoder._extract_barcodes(minimum_pixels=8, maximum_pixels=8)

    spots = decoder.decoded_barcodes.sort_values("gene_id")
    assert spots["gene_id"].tolist() == ["A", "B"]
    np.testing.assert_array_equal(
        spots[["z", "y", "x"]], [[2.5, 3.5, 12.5], [4.5, 7.5, 20.5]]
    )
    # Native calibration, stage origin, camera reflection, then global rotation
    # and translation. Exported coordinates are rounded to 0.01 micrometer.
    np.testing.assert_allclose(
        spots[["global_z", "global_y", "global_x"]],
        [[12.25, -31.12, -17.88], [13.25, -33.12, -18.88]],
        rtol=0,
        atol=1e-10,
    )
