"""Denoise fused PolyDT image using CAREamics Noise2Void model."""

from pathlib import Path
import numpy as np
from tifffile import imwrite
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from careamics.model_io import load_pretrained


def denoise_polydt(root_path: Path, model_path: Path, output_path: Path | None = None) -> None:
    """Denoise the downsampled fused PolyDT image with Noise2Void.

    Parameters
    ----------
    root_path : Path
        Path to experiment root containing ``qi2labdatastore``.
    model_path : Path
        Path to the pretrained Noise2Void Mouse Nuclei model (``.zip`` or ``.ckpt``).
    output_path : Path, optional
        File to save the denoised image. Defaults to ``denoised_polyDT.tif`` in
        ``root_path``.
    """

    datastore_path = root_path / "qi2labdatastore"
    datastore = qi2labDataStore(datastore_path)

    fused, _, _, _ = datastore.load_global_fidicual_image(return_future=False)
    fused = np.asarray(fused, dtype=np.float32)
    fused = np.squeeze(fused)

    careamist, _ = load_pretrained(model_path)

    input_array = fused[None, None, ...]
    prediction = careamist.predict(
        input_array,
        data_type="array",
        axes="SZYX",
    )
    denoised = np.squeeze(np.concatenate(prediction, axis=0))

    if output_path is None:
        output_path = root_path / "denoised_polyDT.tif"

    imwrite(output_path, denoised.astype(np.float32))


if __name__ == "__main__":
    root_path = Path("/path/to/experiment")
    model_path = Path("/path/to/Noise2Void_Mouse_Nuclei.zip")
    denoise_polydt(root_path, model_path)
