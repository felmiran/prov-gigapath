import os
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import timm
from cp_toolbox.image_processing.slide import Wsi
from gigapath.pipeline import load_tile_encoder_transforms
import multiprocessing as mp

class TileGeneratorDataset(Dataset):
    """
    Do encoding for tiles

    Arguments:
    ----------

    transform : torchvision.transforms.Compose
        Transform to apply to each image
    """

    def __init__(self, wsi, tile_size, resolution, coords=None, transform=None):
        self.transform = transform
        self.wsi = wsi  # openslide.OpenSlide object
        self.coords = coords  # np.array of [x, y] coords at OpenSlide level 0
        self.tile_size = tile_size
        self.resolution = resolution
        self.level = wsi.get_level_from_resolution(resolution)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        img = self.wsi.read_region((x, y), self.level, (self.tile_size, self.tile_size)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {'img': torch.from_numpy(np.array(img)),
                'coords': torch.from_numpy(np.array([x, y])).float()}


@torch.no_grad()
def run_inference_with_tile_encoder(tile_encoder: torch.nn.Module, tile_dl) -> dict:
    """
    Run inference with the tile encoder

    Arguments:
    ----------
    tile_dl: tile dataloader
    tile_encoder : torch.nn.Module
        Tile encoder model
    """
    tile_encoder = tile_encoder.cuda()
    # make the tile dataloader
    # tile_dl = DataLoader(TileEncodingDataset(image_paths, transform=load_tile_encoder_transforms()), batch_size=batch_size, shuffle=False)
    # run inference
    tile_encoder.eval()
    collated_outputs = {'features': [], 'coords': []}
    batches = []
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(tile_dl, desc='Running inference with tile encoder'):
            batches.append(batch)
            collated_outputs['features'].append(tile_encoder(batch['img'].cuda()).detach().cpu())
            collated_outputs['coords'].append(batch['coords'])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}


def load_tile_encoder():
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

def encode_wsi_tiles(resolution, batch_size, tile_size, coords_path, wsi_path, save_path, tile_encoder):
    wsi = Wsi(wsi_path)
    level = wsi.get_level_from_resolution(resolution)
    with h5py.File(coords_path, "r") as f:
        coords = f["coords"][()]  # returns as a numpy array

    tile_dl = DataLoader(
        TileGeneratorDataset(
            transform=load_tile_encoder_transforms(),
            wsi=wsi,
            coords=coords,
            tile_size=tile_size,
            resolution=resolution
        ),
        batch_size=batch_size,
        shuffle=False
    )
    output = run_inference_with_tile_encoder(tile_encoder=tile_encoder, tile_dl=tile_dl)
    output["coords"] = output["coords"]  / (2 ** level)  # output coords at selected resolution
    Path(save_path).mkdir(exist_ok=True, parents=True)
    f = h5py.File(os.path.join(save_path, f'{wsi.filename}.h5'), 'w')
    # for key in output:
    #     f.create_dataset(key, data=output[key])
    f.create_dataset("coords", data=output["coords"])
    f.create_dataset("features", data=output["features"])
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser function to encode wsi tiles")
    parser.add_argument("--conf", type=str, help="path to config file", default="tile_encoder_config.json")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        conf = json.load(f)
    f.close()

    print(conf)

    resolution = conf["resolution"]
    batch_size = conf["batch_size"]
    tile_size = conf["tile_size"]
    wsi_list_path = conf["wsi_list_path"]
    coords_dir = conf["coords_dir"]
    wsi_dir = conf["wsi_dir"]
    wsi_path_ext = conf["wsi_path_ext"]
    save_path = conf["save_path"]

    tile_encoder = load_tile_encoder()

    df = pd.read_csv(wsi_list_path)
    wsi_names = df.slide_id
    for wsi_n in wsi_names:
        wsi_path = os.path.join(wsi_dir, wsi_n + wsi_path_ext)
        coords_path = os.path.join(coords_dir, wsi_n + ".h5")
        print(wsi_path, coords_path)



        encode_wsi_tiles(
            resolution=resolution,
            batch_size=batch_size,
            tile_size=tile_size,
            coords_path=coords_path,
            wsi_path=wsi_path,
            save_path=save_path,
            tile_encoder=tile_encoder
        )
