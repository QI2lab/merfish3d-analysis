from micro_sam.sam_annotator import annotator_3d
from micro_sam.sample_data import fetch_3d_example_data
from micro_sam.util import get_cache_directory
from pathlib import Path
import zarr
import numpy as np

EMBEDDING_CACHE = Path("/home/qi2lab/Documents/github/micro-sam") / Path("embeddings")

def lm_3d_annotator(im_data,use_finetuned_model):
    """Run the 3d annotator for a LM volume."""

    if use_finetuned_model:
        embedding_path = EMBEDDING_CACHE / Path("tile1_embeddings-vit_b_lm")
        model_type = "vit_b_lm"
    else:
        embedding_path = EMBEDDING_CACHE / Path("embeddings")
        model_type = "vit_h"

    # start the annotator, cache the embeddings
    annotator_3d(im_data, str(embedding_path), model_type=model_type)


def main():

    use_finetuned_model = True
    
    output_dir_path = Path('/mnt/opm3/20240124_OB_Full_MERFISH_UA_3_allrds/processed_v2')
    polyDT_output_dir_path = output_dir_path / Path('polyDT')

    tile_ids = [entry.name for entry in polyDT_output_dir_path.iterdir() if entry.is_dir()]
    polyDT_current__path = polyDT_output_dir_path / Path(tile_ids[1]) / Path("round000.zarr")
    polyDT_current_tile = zarr.open(polyDT_current__path,mode='r')
    im_data = np.asarray(polyDT_current_tile['raw_data'],dtype=np.uint16)

    lm_3d_annotator(im_data,use_finetuned_model)


if __name__ == "__main__":
    main()