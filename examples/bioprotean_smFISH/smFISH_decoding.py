from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.PixelDecoder import PixelDecoder
from pathlib import Path
# import numpy as np

root_path = Path(r"/data/smFISH/02202025_Bartelle_control_smFISH_TqIB")

# initialize datastore
datastore_path = root_path / Path(r"qi2labdatastore")
datastore = qi2labDataStore(datastore_path)
merfish_bits = datastore.num_bits

# initialize decodor class
decoder = PixelDecoder(
    datastore=datastore, 
    use_mask=False, 
    merfish_bits=merfish_bits, 
    verbose=1,
    smFISH = True
)
# print(f"The distance threshold is {decoder._distance_threshold}")
# print(f"The lower magnitude threshold is {decoder._magnitude_threshold}")
# print(f"The upper magnitude threshold is {decoder._upper_magnitude_threshold}")

# decode one tile
decoder.decode_one_tile(
    tile_idx=0,  # Specify the tile index
    display_results=True,  # Set to True to visualize results in Napari
    lowpass_sigma=(3, 1, 1),  # Lowpass filter sigma
    magnitude_threshold=0.2,  # L2-norm threshold
    upper_magnitude_threshold=1.25,  # Upper L2-norm threshold
    minimum_pixels=3.0,  # Minimum number of pixels for a barcode
    use_normalization=True,  # Use normalization
    ufish_threshold=0.5  # Ufish threshold
)

print("Decoding complete.")

# # Save barcodes
# decoder._save_barcodes()