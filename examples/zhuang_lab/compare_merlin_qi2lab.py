from pathlib import Path
import pandas as pd
from merfish3danalysis.qi2labDataStore import qi2labDataStore
import napari

zhuang_lab_path = Path("/mnt/data/zhuang/mop/mouse_sample1_raw/zhuang_decoded_codewords/spots_mouse1sample1.csv")
datastore_path = Path("/mnt/data/zhuang/mop/mouse_sample1_raw/processed_v3")

datastore = qi2labDataStore(datastore_path)

zhuang_spots_df = pd.read_csv(zhuang_lab_path)
qi2lab_spots_df = datastore.load_global_filtered_decoded_spots()

print(len(zhuang_spots_df))
print(len(qi2lab_spots_df))

coord_columns=['global_y','global_x']
acta2_zhuang_spots_df = zhuang_spots_df[zhuang_spots_df['target_molecule_name'] == 'Acta2'][coord_columns]
coord_columns=['global_x','global_y']
acta2_qi2lab_spots_df = qi2lab_spots_df[qi2lab_spots_df['gene_id'] == 'Acta2'][coord_columns]


viewer = napari.Viewer()
viewer.add_points(acta2_zhuang_spots_df.values,face_color='white')
viewer.add_points(qi2lab_spots_df[coord_columns].values,face_color='red')
napari.run()