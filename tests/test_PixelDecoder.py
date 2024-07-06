import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
from wf_merfish.postprocess.PixelDecoder import PixelDecoder

@pytest.fixture
def mock_codebook():
    return np.random.randint(0, 2, (100, 16))

@pytest.fixture
def mock_image_data():
    return np.random.rand(16, 256, 256)

@pytest.fixture
def mock_pixel_decoder(mock_codebook, mock_image_data):
    with patch('pixel_decoder.PixelDecoder._parse_dataset'), \
         patch('pixel_decoder.PixelDecoder._load_experiment_parameters'), \
         patch('pixel_decoder.PixelDecoder._load_codebook'), \
         patch('pixel_decoder.PixelDecoder._normalize_codebook', return_value=mock_codebook):

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
    
def test_parse_dataset(mock_pixel_decoder):
    # Ensure _parse_dataset method runs without errors
    mock_pixel_decoder._parse_dataset()

def test_load_experiment_parameters(mock_pixel_decoder):
    # Ensure _load_experiment_parameters method runs without errors
    mock_pixel_decoder._load_experiment_parameters()

def test_load_codebook(mock_pixel_decoder):
    # Ensure _load_codebook method runs without errors
    mock_pixel_decoder._load_codebook()

def test_normalize_codebook(mock_pixel_decoder):
    # Ensure _normalize_codebook method returns correct shape
    result = mock_pixel_decoder._normalize_codebook()
    assert result.shape == (100, 16)

def test_lp_filter(mock_pixel_decoder):
    # Ensure _lp_filter method processes image data correctly
    mock_pixel_decoder._lp_filter(sigma=(2, 1, 1))
    assert hasattr(mock_pixel_decoder, '_image_data_lp')

def test_scale_pixel_traces(mock_pixel_decoder):
    # Ensure _scale_pixel_traces method scales pixel traces correctly
    pixel_traces = np.random.rand(16, 256)
    background_vector = np.random.rand(16)
    normalization_vector = np.random.rand(16)
    result = PixelDecoder._scale_pixel_traces(pixel_traces, background_vector, normalization_vector)
    assert result.shape == (16, 256)

def test_clip_pixel_traces(mock_pixel_decoder):
    # Ensure _clip_pixel_traces method clips pixel traces correctly
    pixel_traces = np.random.rand(16, 256)
    result = PixelDecoder._clip_pixel_traces(pixel_traces)
    assert result.shape == (16, 256)

def test_normalize_pixel_traces(mock_pixel_decoder):
    # Ensure _normalize_pixel_traces method normalizes pixel traces correctly
    pixel_traces = np.random.rand(16, 256)
    normalized_traces, norms = PixelDecoder._normalize_pixel_traces(pixel_traces)
    assert normalized_traces.shape == (16, 256)
    assert norms.shape == (256,)

def test_calculate_distances(mock_pixel_decoder):
    # Ensure _calculate_distances method calculates distances correctly
    pixel_traces = np.random.rand(16, 256)
    codebook_matrix = np.random.rand(100, 16)
    distances, indices = PixelDecoder._calculate_distances(pixel_traces, codebook_matrix)
    assert distances.shape == (256,)
    assert indices.shape == (256,)

def test_decode_pixels(mock_pixel_decoder):
    # Ensure _decode_pixels method processes pixel data correctly
    mock_pixel_decoder._decode_pixels()
    assert hasattr(mock_pixel_decoder, '_decoded_image')
    assert hasattr(mock_pixel_decoder, '_magnitude_image')
    assert hasattr(mock_pixel_decoder, '_scaled_pixel_images')
    assert hasattr(mock_pixel_decoder, '_distance_image')

def test_extract_barcodes(mock_pixel_decoder):
    # Ensure _extract_barcodes method processes decoded images correctly
    mock_pixel_decoder._extract_barcodes()
    assert hasattr(mock_pixel_decoder, '_df_barcodes')

def test_save_barcodes(mock_pixel_decoder):
    # Mock saving barcodes to avoid file I/O
    with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
        mock_pixel_decoder._save_barcodes()
        assert mock_to_csv.called or mock_to_parquet.called

def test_cleanup(mock_pixel_decoder):
    # Ensure _cleanup method runs without errors
    mock_pixel_decoder._cleanup()

def test_decode_one_tile(mock_pixel_decoder):
    # Mock methods that involve file I/O
    with patch('pixel_decoder.PixelDecoder._load_global_normalization_vectors'), \
         patch('pixel_decoder.PixelDecoder._global_normalization_vectors'), \
         patch('pixel_decoder.PixelDecoder._load_bit_data'), \
         patch('pixel_decoder.PixelDecoder._lp_filter'), \
         patch('pixel_decoder.PixelDecoder._decode_pixels'), \
         patch('pixel_decoder.PixelDecoder._display_results'), \
         patch('pixel_decoder.PixelDecoder._cleanup'):
        mock_pixel_decoder.decode_one_tile(tile_idx=0, display_results=False)