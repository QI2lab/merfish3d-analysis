from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
from typing import Union, Optional, Sequence, Collection
import pandas as pd
from itertools import product

root_path = Path(r"/data/smFISH/02202025_Bartelle_control_smFISH_TqIB")

# Initialize qi2labDataStore.
class qi2labDataStore:
    def __init__(self, datastore_path: Union[str, Path]):
        self._datastore_path = Path(datastore_path)
        self._ufish_localizations_root_path = self._datastore_path / Path(r"ufish_localizations")
        # print(self._ufish_localizations_root_path)

    def read_ufish_localizations(self, tile_id, bit_id):
        # this method reads the Parquet file located at the constructed path (tile0000/bit001.parquet) and returns its contents as a pandas DataFrame
        current_ufish_localizations_path = (
            self._ufish_localizations_root_path
            / Path(tile_id)
            / Path(bit_id + ".parquet")
        )
        # if not current_ufish_localizations_path.exists():
        #     print("U-FISH localizations not found.")
        #  print(f"Attempting to read file: {current_ufish_localizations_path}")
        return pd.read_parquet(current_ufish_localizations_path)
    
    def get_tile_ids(self):
        # Retrieve tile IDs from the directory structure
        return sorted([p.name for p in self._ufish_localizations_root_path.iterdir() if p.is_dir()], key=lambda x: int(x.replace("tile", "")))

    def get_bit_ids(self, tile_id):
        # Retrieve bit IDs from the directory structure of a specific tile
        tile_path = self._ufish_localizations_root_path / Path(tile_id)
        return sorted([p.stem for p in tile_path.glob("*.parquet")], key=lambda x: int(x.replace("bit", "")))

datastore_path = root_path / Path(r"qi2labdatastore")
# initialize the qi2labDataStore class
datastore = qi2labDataStore(datastore_path)


# Use the methods to get tile_ids and bit_ids
tile_ids = datastore.get_tile_ids()
for tile_id in tile_ids:
    bit_ids = datastore.get_bit_ids(tile_id)
    for tile_id, bit_id in product([tile_id], bit_ids):
        ufish_localizations_df = datastore.read_ufish_localizations(tile_id, bit_id)

        # Set the threshold for the ufish probabilities
        threshold = 5

        # grab the ufish localizations
        ufish_probabilities = ufish_localizations_df.loc[:, "sum_prob_pixels"]
        # set everything <5 = 0
        thresholded_probabilities = []
        for prob in ufish_probabilities:
            if prob < int(threshold):
                thresholded_probabilities.append(0)
            else:
                thresholded_probabilities.append(prob)

        # Create a new column with thresholded values
        ufish_localizations_df["thresholded_probabilities"] = thresholded_probabilities

        # Save the updated DataFrame back to a new Parquet file
        output_path = datastore_path / "ufish_localizations_thresholded" / tile_id / f"{bit_id}_thresholded.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
        ufish_localizations_df.to_parquet(output_path)

print(f"Thresholded probabilities saved to {output_path}")
