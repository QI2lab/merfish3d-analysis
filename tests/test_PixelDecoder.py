import pytest
import numpy as np
import cupy as cp
import pandas as pd
from pathlib import Path
from wf_merfish.postprocess.PixelDecoder import PixelDecoder

@pytest.fixture
def mock_codebook():
    return [
        ['genes001', 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ['genes002', 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        ['genes003', 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ['genes004', 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ['genes005', 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ['genes006', 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        ['genes007', 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ['genes008', 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
        ['genes009', 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ['genes010', 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        ['Blank01', 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ['Blank02', 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        ['Blank03', 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        ['Blank04', 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        ['Blank05', 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ]
    

@pytest.fixture
def mock_image_data():
    return np.random.rand(16, 256, 256)

@pytest.fixture
def mock_tile_ids():
    return ['tile001', 'tile002', 'tile003', 'tile004', 'tile005']

@pytest.fixture
def mock_bit_ids():
    return ['bit01', 'bit02', 'bit03', 'bit04', 'bit05', 'bit06', 'bit07', 'bit08', 'bit09', 'bit10', 'bit11', 'bit12', 'bit13', 'bit14', 'bit15', 'bit16']


@pytest.fixture
def mock_pixel_decoder(mock_codebook, mock_image_data,mocker):
    with mocker.patch('wf_merfish.postprocess.PixelDecoder._parse_dataset'), \
         mocker.patch('wf_merfish.postprocess.PixelDecoder._load_experiment_parameters'), \
         mocker.patch('wf_merfish.postprocess.PixelDecoder._load_codebook'), \
         mocker.patch('wf_merfish.postprocess.PixelDecoder._normalize_codebook', return_value=mock_codebook):

        decoder = PixelDecoder(
            dataset_path='mock/path/to/zarr',
            exp_type='3D',
            use_mask=False,
            z_range=None,
            include_blanks=True,
            merfish_bits=16,
            verbose=1
        )
        
        decoder._codebook_matrix = mock_codebook
        decoder._image_data = mock_image_data
        decoder._df_codebook = pd.DataFrame(mock_codebook)
        
        return decoder
    
def create_dataset_structure(base_path, mock_tile_ids, mock_bit_ids):
    readouts_path = base_path / 'readouts'
    readouts_path.mkdir(parents=True, exist_ok=True)
    
    tile_ids = mock_tile_ids
    bit_ids = mock_bit_ids

    # Create tile directories
    for tile_id in tile_ids:  # Assuming 3 tiles for this example
        tile_path = readouts_path / Path(tile_id+'.zarr')
        tile_path.mkdir(parents=True, exist_ok=True)
        
        # Create bit directories
        for bit_id in bit_ids:  # Assuming 16 bits
            bit_path = tile_path / Path(bit_id+'.zarr')
            bit_path.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def temp_dataset(tmp_path, mock_tile_ids, mock_bit_ids):
    create_dataset_structure(tmp_path, mock_tile_ids, mock_bit_ids)
    return tmp_path

def test_parse_dataset(temp_dataset, mocker, mock_tile_ids, mock_bit_ids):
    
    mocker.patch.object(PixelDecoder, "_load_experiment_parameters", autospec=True)
    mocker.patch.object(PixelDecoder, "_load_codebook", autospec=True)
    mocker.patch.object(PixelDecoder, "_normalize_codebook", return_value=np.random.rand(100, 16), autospec=True)
    
    # Initialize the PixelDecoder with the temp_dataset path
    decoder = PixelDecoder(
        dataset_path=temp_dataset,
        exp_type='3D',
        use_mask=False,
        z_range=None,
        include_blanks=True,
        merfish_bits=16,
        verbose=1
    )
        
    # Assertions
    expected_tile_ids = mock_tile_ids
    expected_bit_ids = mock_bit_ids
    
    assert decoder._tile_ids == expected_tile_ids
    assert decoder._bit_ids == expected_bit_ids
    
def test_load_experiment_parameters(temp_dataset, mocker):
    
    mocker.patch.object(PixelDecoder, "_load_experiment_parameters", autospec=True)
    mocker.patch.object(PixelDecoder, "_load_codebook", autospec=True)
    mocker.patch.object(PixelDecoder, "_normalize_codebook", return_value=np.random.rand(100, 16), autospec=True)
    
    # Initialize the PixelDecoder with the temp_dataset path
    decoder = PixelDecoder(
        dataset_path=temp_dataset,
        exp_type='3D',
        use_mask=False,
        z_range=None,
        include_blanks=True,
        merfish_bits=16,
        verbose=1
    )
        
    # Assertions
    expected_tile_ids = ['tile0.zarr', 'tile1.zarr', 'tile2.zarr']
    expected_bit_ids = [f'bit{i}.zarr' for i in range(16)]
    
    assert decoder._tile_ids == expected_tile_ids
    assert decoder._bit_ids == expected_bit_ids
    

def test_load_experiment_parameters(temp_dataset, mocker):
    # Mock zarr.open to avoid actual file I/O
    mock_zarr_open = mocker.patch('zarr.open', autospec=True)
    
    mocker.patch.object(PixelDecoder, "_parse_dataset", autospec=True)
    mocker.patch.object(PixelDecoder, "_load_codebook", autospec=True)
    mocker.patch.object(PixelDecoder, "_normalize_codebook", return_value=np.random.rand(100, 16), autospec=True)
    
    # Initialize the PixelDecoder with the temp_dataset path
    decoder = PixelDecoder(
        dataset_path=temp_dataset,
        exp_type='3D',
        use_mask=False,
        z_range=None,
        include_blanks=True,
        merfish_bits=16,
        verbose=1
    )
    
    # Assertions
    expected_path = temp_dataset / "calibrations.zarr"
    mock_zarr_open.assert_called_once_with(expected_path, mode='a')
    
    assert decoder._na == 1.35
    assert decoder._ri == 1.4
    assert decoder._num_on_bits == 4

@pytest.mark.parametrize("include_blanks,expected_df,expected_codebook_matrix,expected_gene_ids,expected_blank_count", [
    (True, 
     pd.DataFrame([
        ['genes001', 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ['genes002', 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        ['genes003', 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ['genes004', 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ['genes005', 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ['genes006', 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        ['genes007', 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ['genes008', 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
        ['genes009', 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ['genes010', 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        ['Blank01', 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ['Blank02', 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        ['Blank03', 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        ['Blank04', 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        ['Blank05', 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=object).astype({1: int, 2: int, 3: int, 4: int, 5: int, 6: int,
                             7: int, 8: int, 9: int, 10: int, 11: int, 12: int,
                             13: int, 14: int, 15: int, 16: int, 17: int, 18: int}),
     np.array([
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ]),
     ['genes001', 'genes002', 'genes003', 'genes004', 'genes005', 'genes006',
      'genes007', 'genes008', 'genes009', 'genes010', 'Blank01', 'Blank02',
      'Blank03', 'Blank04', 'Blank05'],
     5),
    (False, 
     pd.DataFrame([
        ['genes001', 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ['genes002', 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        ['genes003', 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ['genes004', 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ['genes005', 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ['genes006', 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        ['genes007', 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ['genes008', 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
        ['genes009', 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ['genes010', 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    ], dtype=object).astype({1: int, 2: int, 3: int, 4: int, 5: int, 6: int,
                             7: int, 8: int, 9: int, 10: int, 11: int, 12: int,
                             13: int, 14: int, 15: int, 16: int, 17: int, 18: int}),
     np.array([
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    ]),
     ['genes001', 'genes002', 'genes003', 'genes004', 'genes005', 'genes006',
      'genes007', 'genes008', 'genes009', 'genes010'],
     5)
])
def test_load_codebook(temp_dataset, mocker, mock_codebook, include_blanks, expected_df, expected_codebook_matrix, expected_gene_ids, expected_blank_count):
    # Mock _calibration_zarr to include the known codebook array
    mock_calibration_zarr = mocker.MagicMock()
    mock_calibration_zarr.attrs = {'codebook': mock_codebook}
    
    # Patch zarr.open to return the mocked _calibration_zarr
    mocker.patch('zarr.open', return_value=mock_calibration_zarr)
    
    # Patch the PixelDecoder methods to avoid actual file I/O
    mocker.patch.object(PixelDecoder, "_parse_dataset", autospec=True)
    mocker.patch.object(PixelDecoder, "_normalize_codebook", autospec=True)
    
    # Initialize the PixelDecoder with the temp_dataset path
    decoder = PixelDecoder(
        dataset_path=temp_dataset,
        exp_type='3D',
        use_mask=False,
        z_range=None,
        include_blanks=include_blanks,
        merfish_bits=16,
        verbose=1
    )
    
    # Assertions
    pd.testing.assert_frame_equal(decoder._df_codebook.reset_index(drop=True), expected_df.reset_index(drop=True))
    assert np.array_equal(decoder._codebook_matrix, expected_codebook_matrix)
    assert decoder._gene_ids == expected_gene_ids
    assert decoder._blank_count == expected_blank_count

def calculate_expected_normalized_codebook(mock_codebook, n_positive_bits, merfish_bits):
    codebook_array = np.array([row[1:merfish_bits+1] for row in mock_codebook], dtype=float)
    normalized_codebook = codebook_array / np.sqrt(n_positive_bits)
    return cp.asarray(normalized_codebook)

@pytest.mark.parametrize("include_errors", [False])
def test_normalize_codebook(temp_dataset, mocker, mock_codebook, include_errors):
    n_positive_bits = 4  # Example value, adjust as needed
    expected_normalized_codebook = calculate_expected_normalized_codebook(mock_codebook, n_positive_bits, 16)

    # Mock _calibration_zarr to include the known codebook array
    mock_calibration_zarr = mocker.MagicMock()
    mock_calibration_zarr.attrs = {'codebook': mock_codebook}

    # Patch the PixelDecoder methods to avoid actual file I/O
    mocker.patch.object(PixelDecoder, "_parse_dataset", autospec=True)
    mocker.patch.object(PixelDecoder, "_load_codebook", autospec=True, side_effect=lambda self: setattr(self, "_codebook_matrix", expected_normalized_codebook))

    # Patch zarr.open to return the mocked _calibration_zarr
    mocker.patch('zarr.open', return_value=mock_calibration_zarr)

    # Initialize the PixelDecoder with the temp_dataset path
    decoder = PixelDecoder(
        dataset_path=temp_dataset,
        exp_type='3D',
        use_mask=False,
        z_range=None,
        include_blanks=True,  # or False, depending on what you need for your test
        merfish_bits=16,
        verbose=1
    )

    # Normalize the codebook
    normalized_codebook = decoder._normalize_codebook(include_errors)
    
    # Compare the results
    np.testing.assert_almost_equal(cp.asnumpy(normalized_codebook), cp.asnumpy(expected_normalized_codebook))
    
    
@pytest.fixture
def mock_normalization_vectors():
    return {
        'global_normalization': [0.5] * 16,
        'global_background': [0.1] * 16
    }

def test_load_global_normalization_vectors_success(temp_dataset, mocker, mock_normalization_vectors):
    # Mock _calibration_zarr to include the known normalization vectors
    mock_calibration_zarr = mocker.MagicMock()
    mock_calibration_zarr.attrs = mock_normalization_vectors

    # Patch the PixelDecoder methods to avoid actual file I/O
    mocker.patch.object(PixelDecoder, "_parse_dataset", autospec=True)
    mocker.patch.object(PixelDecoder, "_load_codebook", autospec=True)
    mocker.patch.object(PixelDecoder, "_normalize_codebook", autospec=True)

    # Patch zarr.open to return the mocked _calibration_zarr
    mocker.patch('zarr.open', return_value=mock_calibration_zarr)
    # Patch cp.asarray to return a cupy array from the mock normalization vectors
    mocker.patch('cupy.asarray', side_effect=lambda x: cp.array(x))

    # Initialize the PixelDecoder with the temp_dataset path
    decoder = PixelDecoder(
        dataset_path=temp_dataset,
        exp_type='3D',
        use_mask=False,
        z_range=None,
        include_blanks=True,
        merfish_bits=16,
        verbose=1
    )

    # Call the method
    decoder._load_global_normalization_vectors()

    # Assertions
    assert decoder._global_normalization_loaded is True
    assert cp.array_equal(decoder._global_background_vector, cp.array(mock_normalization_vectors['global_background']))
    assert cp.array_equal(decoder._global_normalization_vector, cp.array(mock_normalization_vectors['global_normalization']))

def test_load_global_normalization_vectors_fallback(temp_dataset, mocker):
    # Mock _calibration_zarr without the normalization vectors
    mock_calibration_zarr = mocker.MagicMock()
    mock_calibration_zarr.attrs = {}

    # Patch the PixelDecoder methods to avoid actual file I/O
    mocker.patch.object(PixelDecoder, "_parse_dataset", autospec=True)
    mocker.patch.object(PixelDecoder, "_load_codebook", autospec=True)
    mocker.patch.object(PixelDecoder, "_normalize_codebook", autospec=True)

    # Patch zarr.open to return the mocked _calibration_zarr
    mocker.patch('zarr.open', return_value=mock_calibration_zarr)
    # Patch cp.asarray to raise an exception when trying to access missing attributes
    mocker.patch('cupy.asarray', side_effect=KeyError)

    # Patch _global_normalization_vectors to check if it's called
    mock_global_normalization_vectors = mocker.patch.object(PixelDecoder, "_global_normalization_vectors", autospec=True)

    # Initialize the PixelDecoder with the temp_dataset path
    decoder = PixelDecoder(
        dataset_path=temp_dataset,
        exp_type='3D',
        use_mask=False,
        z_range=None,
        include_blanks=True,
        merfish_bits=16,
        verbose=1
    )

    # Call the method
    decoder._load_global_normalization_vectors()

    # Assertions
    assert mock_global_normalization_vectors.called

# Function to generate matrices with known percentiles
def generate_matrices_with_known_percentiles():
    shape = (3, 512, 512)
    num_tiles = 5
    num_bits = 16

    matrices = np.zeros((num_tiles, num_bits, *shape), dtype=np.float32)
    low_value = 150
    high_value = 1000
    noise_sigma = 50

    for bit_idx in range(num_bits):
        for tile_idx in range(num_tiles):
            matrix = np.zeros(shape, dtype=np.float32)

            # Add low value to all pixels
            matrix += low_value

            # Add high value to 30% of pixels
            high_indices = np.random.choice(matrix.size, size=int(matrix.size * 0.3), replace=False)
            matrix.flat[high_indices] += high_value

            # Add Gaussian noise
            matrix += np.random.normal(0, noise_sigma, size=matrix.shape)

            matrices[tile_idx, bit_idx, :] = matrix

    # Calculate expected percentiles
    expected_background_vector = np.percentile(matrices, 10, axis=(0, 2, 3))
    expected_normalization_vector = np.percentile(matrices, 90, axis=(0, 2, 3))
    
    return matrices, expected_background_vector, expected_normalization_vector

@pytest.fixture
def static_image_data():
    return generate_matrices_with_known_percentiles()

def test_global_normalization_vectors(temp_dataset, mocker, static_image_data, mock_tile_ids, mock_bit_ids):
    images, expected_background_vector, expected_normalization_vector = static_image_data

    # Mock necessary PixelDecoder attributes and methods
    mock_calibration_zarr = mocker.MagicMock()
    mock_calibration_zarr.attrs = {}

    mocker.patch.object(PixelDecoder, "_parse_dataset", autospec=True)
    mocker.patch.object(PixelDecoder, "_load_codebook", autospec=True)
    mocker.patch.object(PixelDecoder, "_normalize_codebook", autospec=True)
    mocker.patch('zarr.open', return_value=mock_calibration_zarr)

    decoder = PixelDecoder(
        dataset_path=temp_dataset,
        exp_type='3D',
        use_mask=False,
        z_range=None,
        include_blanks=True,
        merfish_bits=16,
        verbose=1
    )

    decoder._tile_ids = mock_tile_ids
    decoder._bit_ids = mock_bit_ids
    decoder._readout_dir_path = Path("/mock/path/to/readouts")

    def mock_zarr_open(path, mode='r'):
        class MockZarr:
            def __init__(self, data):
                self.data = data
                self.attrs = {
                    'registered_ufish_data': data,
                    'registered_decon_data': data
                }
                
            def __getitem__(self, key):
                return self.data

        tile_idx = int(path.parts[-2].split('tile')[-1]) - 1
        bit_idx = int(path.parts[-1].split('bit')[-1]) - 1
        return MockZarr(images[tile_idx, bit_idx, :])

    mocker.patch('zarr.open', mock_zarr_open)

    decoder._global_normalization_vectors()
    
    assert decoder._global_normalization_loaded is True
    np.testing.assert_almost_equal(cp.asnumpy(decoder._global_background_vector), expected_background_vector, decimal=6)
    np.testing.assert_almost_equal(cp.asnumpy(decoder._global_normalization_vector), expected_normalization_vector, decimal=6)


        
# def test_lp_filter(mock_pixel_decoder):
#     # Ensure _lp_filter method processes image data correctly
#     mock_pixel_decoder._lp_filter(sigma=(2, 1, 1))
#     assert hasattr(mock_pixel_decoder, '_image_data_lp')

# def test_scale_pixel_traces(mock_pixel_decoder):
#     # Ensure _scale_pixel_traces method scales pixel traces correctly
#     pixel_traces = np.random.rand(16, 256)
#     background_vector = np.random.rand(16)
#     normalization_vector = np.random.rand(16)
#     result = PixelDecoder._scale_pixel_traces(pixel_traces, background_vector, normalization_vector)
#     assert result.shape == (16, 256)

# def test_clip_pixel_traces(mock_pixel_decoder):
#     # Ensure _clip_pixel_traces method clips pixel traces correctly
#     pixel_traces = np.random.rand(16, 256)
#     result = PixelDecoder._clip_pixel_traces(pixel_traces)
#     assert result.shape == (16, 256)

# def test_normalize_pixel_traces(mock_pixel_decoder):
#     # Ensure _normalize_pixel_traces method normalizes pixel traces correctly
#     pixel_traces = np.random.rand(16, 256)
#     normalized_traces, norms = PixelDecoder._normalize_pixel_traces(pixel_traces)
#     assert normalized_traces.shape == (16, 256)
#     assert norms.shape == (256,)

# def test_calculate_distances(mock_pixel_decoder):
#     # Ensure _calculate_distances method calculates distances correctly
#     pixel_traces = np.random.rand(16, 256)
#     codebook_matrix = np.random.rand(100, 16)
#     distances, indices = PixelDecoder._calculate_distances(pixel_traces, codebook_matrix)
#     assert distances.shape == (256,)
#     assert indices.shape == (256,)

# def test_decode_pixels(mock_pixel_decoder):
#     # Ensure _decode_pixels method processes pixel data correctly
#     mock_pixel_decoder._decode_pixels()
#     assert hasattr(mock_pixel_decoder, '_decoded_image')
#     assert hasattr(mock_pixel_decoder, '_magnitude_image')
#     assert hasattr(mock_pixel_decoder, '_scaled_pixel_images')
#     assert hasattr(mock_pixel_decoder, '_distance_image')

# def test_extract_barcodes(mock_pixel_decoder):
#     # Ensure _extract_barcodes method processes decoded images correctly
#     mock_pixel_decoder._extract_barcodes()
#     assert hasattr(mock_pixel_decoder, '_df_barcodes')

# def test_save_barcodes(mock_pixel_decoder,mocker):
#     # Mock saving barcodes to avoid file I/O
#     with mocker.patch('pandas.DataFrame.to_csv') as mock_to_csv, \
#          mocker.patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
#         mock_pixel_decoder._save_barcodes()
#         assert mock_to_csv.called or mock_to_parquet.called

# def test_cleanup(mock_pixel_decoder):
#     # Ensure _cleanup method runs without errors
#     mock_pixel_decoder._cleanup()

# def test_decode_one_tile(mock_pixel_decoder,mocker):
#     # Mock methods that involve file I/O
#     with mocker.patch('pixel_decoder.PixelDecoder._load_global_normalization_vectors'), \
#          mocker.patch('pixel_decoder.PixelDecoder._global_normalization_vectors'), \
#          mocker.patch('pixel_decoder.PixelDecoder._load_bit_data'), \
#          mocker.patch('pixel_decoder.PixelDecoder._lp_filter'), \
#          mocker.patch('pixel_decoder.PixelDecoder._decode_pixels'), \
#          mocker.patch('pixel_decoder.PixelDecoder._display_results'), \
#          mocker.patch('pixel_decoder.PixelDecoder._cleanup'):
#         mock_pixel_decoder.decode_one_tile(tile_idx=0, display_results=False)