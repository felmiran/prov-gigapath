import os
import argparse
import json
import torch
import h5py
from pathlib import Path
import numpy as np

from cp_toolbox.utils.utils import list_file_paths

import gigapath.slide_encoder
from gigapath.pipeline import run_inference_with_slide_encoder


def encode_slides(tile_embedding_paths, slide_encoder, save_path):
    # TODO: make this into a Dataset for embeddings. Easy to do.
    for i, p in enumerate(tile_embedding_paths):
        print(f"processing slide {i+1}: {Path(p).stem}")
        with h5py.File(p, "r") as f:
            coords = f["coords"][()]  # returns as a numpy array
            tile_embeds = f["features"][()]  # returns as a numpy array

        output = run_inference_with_slide_encoder(
            tile_embeds=torch.tensor(tile_embeds),
            coords=torch.tensor(coords),
            slide_encoder_model=slide_encoder
        )

        Path(save_path).mkdir(exist_ok=True, parents=True)
        with h5py.File(os.path.join(save_path, f'{Path(p).stem}.h5'), 'w') as f:
            for o in output:
                f.create_dataset(o, data=np.squeeze(output[o].numpy()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser function to encode WSI tiles")
    parser.add_argument("--conf", type=str, help="path to config file", default="slide_encoder_config.json")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        conf = json.load(f)

    save_path = conf["save_path"]
    tile_embedding_paths = list_file_paths(conf["tile_embedding_paths"], [".h5"])

    slide_encoder = gigapath.slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath",
                                                        "gigapath_slide_enc12l768d", 1536)

    encode_slides(tile_embedding_paths, slide_encoder, save_path)








