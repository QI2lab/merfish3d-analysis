import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from wf_merfish.postprocess.PixelDecoder import PixelDecoder

@pytest.fixture
def mock_codebook():
    return np.random.randint(0, 2, (100, 16))

@pytest.fixture
def mock_image_data():
    return np.random.rand(16, 256, 256)

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
    
def create_dataset_structure(base_path):
    readouts_path = base_path / 'readouts'
    readouts_path.mkdir(parents=True, exist_ok=True)

    # Create tile directories
    for tile_idx in range(3):  # Assuming 3 tiles for this example
        tile_path = readouts_path / f'tile{tile_idx}.zarr'
        tile_path.mkdir(parents=True, exist_ok=True)
        
        # Create bit directories
        for bit_idx in range(16):  # Assuming 16 bits
            bit_path = tile_path / f'bit{bit_idx}.zarr'
            bit_path.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def temp_dataset(tmp_path):
    create_dataset_structure(tmp_path)
    return tmp_path

def test_parse_dataset(temp_dataset, mocker):
    
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

# def test_load_codebook(mock_pixel_decoder):
#     # Ensure _load_codebook method runs without errors
#     mock_pixel_decoder._load_codebook()

# def test_normalize_codebook(mock_pixel_decoder):
#     # Ensure _normalize_codebook method returns correct shape
#     result = mock_pixel_decoder._normalize_codebook()
#     assert result.shape == (100, 16)

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