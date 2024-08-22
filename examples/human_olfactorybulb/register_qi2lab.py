"""Perform registration on Human OB qi2labdatastore.

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.postprocess.DataRegistration import DataRegistration
from pathlib import Path

# root data folder
root_path = Path(r"/mnt/data/qi2lab/20240807_OB_22bit_PL028_2")

# # initialize datastore
datastore_path = root_path / Path(r"qi2labdatastore")
datastore = qi2labDataStore(datastore_path)
registration_factory = DataRegistration(
    datastore=datastore,
    perform_optical_flow=True,
)

registration_factory.register_all_tiles()