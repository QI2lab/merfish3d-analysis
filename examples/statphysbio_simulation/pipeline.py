from merfish3danalysis.cli.statphysbio_simulation.convert_simulation_to_experiment import (
    convert_simulation,
)
from merfish3danalysis.cli.statphysbio_simulation.convert_to_datastore import (
    convert_data,
)
from merfish3danalysis.cli.statphysbio_simulation.register_and_deconvolve import (
    manage_data_registration_states,
)
from merfish3danalysis.cli.statphysbio_simulation.sweep_f1 import (
    sweep_decode_params,
)
from pathlib import Path


def main():
    data_path = Path(r"/home/hblanc01/Data/density_smFISH_flat/Nmols_1500")
    convert_simulation(data_path)
    convert_data(data_path / "sim_acquisition")
    manage_data_registration_states(data_path / "sim_acquisition")
    sweep_decode_params(
        root_path=data_path / "sim_acquisition", gt_path=data_path / "GT_spots.csv"
    )


if __name__ == "__main__":
    main()
