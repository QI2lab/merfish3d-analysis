from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import pandas as pd

root_path = Path(r"/data/smFISH/02202025_Bartelle_control_smFISH_TqIB")

class smFISH_spotdetection:
    def __init__(self, datastore, threshold=5):
        self.datastore = datastore
        self.threshold = threshold

    def read_ufish_localizations(self, tile_id, bit_id):
        self._ufish_localizations_root_path = self.datastore._ufish_localizations_root_path
        current_ufish_localizations_path = (
            self._ufish_localizations_root_path
            / Path(tile_id)
            / Path(bit_id + ".parquet")
        )
        if not current_ufish_localizations_path.exists():
            raise FileNotFoundError(f"File {current_ufish_localizations_path} does not exist.")
        return pd.read_parquet(current_ufish_localizations_path)

    def get_tile_ids(self):
        if not self.datastore._ufish_localizations_root_path.exists():
            raise FileNotFoundError(f"Directory {self.datastore._ufish_localizations_root_path} does not exist.")
        return sorted([p.name for p in self.datastore._ufish_localizations_root_path.iterdir() if p.is_dir()], key=lambda x: int(x.replace("tile", "")))

    def get_bit_ids(self, tile_id):
        tile_path = self.datastore._ufish_localizations_root_path / Path(tile_id)
        if not tile_path.exists():
            raise FileNotFoundError(f"Directory {tile_path} does not exist.")
        return sorted([p.stem for p in tile_path.glob("*.parquet")], key=lambda x: int(x.replace("bit", "")))


datastore_path = root_path / Path(r"qi2labdatastore")
datastore = qi2labDataStore(datastore_path)
smfish_class = smFISH_spotdetection(datastore)

tile_ids = smfish_class.get_tile_ids()
for tile_id in tile_ids:
    bit_ids = smfish_class.get_bit_ids(tile_id)
    for bit_id in bit_ids:
        ufish_localizations_df = smfish_class.read_ufish_localizations(tile_id, bit_id)

        ufish_probabilities = ufish_localizations_df.loc[:, "sum_prob_pixels"]
        ufish_localizations_df["thresholded_probabilities"] = ufish_probabilities.where(ufish_probabilities >= smfish_class.threshold, 0)

        output_path = datastore_path / "ufish_localizations_thresholded" / tile_id / f"{bit_id}_thresholded.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ufish_localizations_df.to_parquet(output_path)

print("Thresholded probabilities saved")