from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
from typing import Union, Optional, Sequence, Collection
import pandas as pd

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

datastore_path = root_path / Path(r"qi2labdatastore")
# initialize the qi2labDataStore class
datastore = qi2labDataStore(datastore_path)

# Read the ufish localizations from the Parquet file
ufish_localizations_df = datastore.read_ufish_localizations("tile0000", "bit001")
# print(ufish_localizations_df)

# Output:
#        z     y     x  sum_prob_pixels  sum_decon_pixels  tile_idx  bit_idx  tile_z_px  tile_y_px  tile_x_px
# 0      0     9  1138        33.477734              6646         0        1          0          9       1138
# 1      0   236   563        17.889059             23476         0        1          0        236        563
# 2      0   295   123        43.437790             11803         0        1          0        295        123
# 3      0   355   307        41.473850            398292         0        1          0        355        307
# 4      0   362  1336        58.301956            355961         0        1          0        362       1336
# ...   ..   ...   ...              ...               ...       ...      ...        ...        ...        ...
# 9009  49  1942  1100        19.608427              2911         0        1         49       1942       1100
# 9010  49  1986  1986        20.223333              2028         0        1         49       1986       1986
# 9011  49  1999    74        21.493113              2139         0        1         49       1999         74
# 9012  49  2014    86        15.593759              1597         0        1         49       2014         86
# 9013  49  2018  1375        25.273598              5003         0        1         49       2018       1375

# [9014 rows x 10 columns]


# Set the threshold for the ufish probabilities
# threshold = 5

# grab the ufish localizations
ufish_probabilities = ufish_localizations_df.loc[:, "sum_prob_pixels"]
# set everything <5 = 0
thresholded_probabilities = []
for prob in ufish_probabilities:
    if prob < 5:
        thresholded_probabilities.append(0)
    else:
        thresholded_probabilities.append(prob)
# print(thresholded_probabilities)
# Create a new column with thresholded values
ufish_localizations_df["thresholded_probabilities"] = thresholded_probabilities


# Save the updated DataFrame back to a new Parquet file
output_path = datastore_path / "ufish_localizations_thresholded" / "tile0000" / "bit001_thresholded.parquet"
output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
ufish_localizations_df.to_parquet(output_path)

print(f"Thresholded probabilities saved to {output_path}")
