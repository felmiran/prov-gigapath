import os
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import timm
from cp_toolbox.image_processing.slide import Wsi
from gigapath.pipeline import load_tile_encoder_transforms
import torch.multiprocessing as mp


class TileGeneratorDataset(Dataset):
    def __init__(self, wsi_path, tile_size, resolution, coords=None, transform=None):
        self.transform = transform
        self.wsi_path = wsi_path  # Pass path, not the object
        self.coords = coords
        self.tile_size = tile_size
        self.resolution = resolution
        self.wsi = None

    def _load_wsi(self):
        if self.wsi is None:  # Load Wsi object when needed
            self.wsi = Wsi(self.wsi_path)
            self.level = self.wsi.get_level_from_resolution(self.resolution)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        self._load_wsi()  # Load Wsi object in the process
        x, y = self.coords[idx]
        img = self.wsi.read_region((x, y), self.level, (self.tile_size, self.tile_size)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            'img': torch.from_numpy(np.array(img)),
            'coords': torch.from_numpy(np.array([x, y])).float() / (2**self.level)
        }


@torch.no_grad()
def run_inference_with_tile_encoder(tile_encoder: torch.nn.Module, tile_dl, gpu_id) -> dict:
    tile_encoder = tile_encoder.cuda(gpu_id)
    tile_encoder.eval()
    collated_outputs = {'features': [], 'coords': []}
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(tile_dl, desc=f'Running inference on GPU {gpu_id}'):
            collated_outputs['features'].append(tile_encoder(batch['img'].cuda(gpu_id)).detach().cpu())
            collated_outputs['coords'].append(batch['coords'])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}


def load_tile_encoder():
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)


def encode_wsi_tiles(resolution, batch_size, tile_size, coords_path, wsi_path, save_path, tile_encoder, gpu_id):
    # Load coords from the HDF5 file
    with h5py.File(coords_path, 'r') as f:
        coords = f['coords'][:]
    
    tile_dl = DataLoader(
        TileGeneratorDataset(
            transform=load_tile_encoder_transforms(),
            wsi_path=wsi_path,
            coords=coords,
            tile_size=tile_size,
            resolution=resolution
        ),
        batch_size=batch_size,
        shuffle=False,
        # num_workers=0  # You may reduce workers to avoid overloading system
    )
    output = run_inference_with_tile_encoder(tile_encoder=tile_encoder, tile_dl=tile_dl, gpu_id=gpu_id)

    Path(save_path).mkdir(exist_ok=True, parents=True)
    with h5py.File(os.path.join(save_path, f'{Path(wsi_path).stem}.h5'), 'w') as f:
        f.create_dataset("coords", data=output["coords"])
        f.create_dataset("features", data=output["features"])


def run_inference_on_all_gpus(gpu_id, conf, tile_encoder):
    torch.cuda.set_device(gpu_id)

    resolution = conf["resolution"]
    batch_size = conf["batch_size"]
    tile_size = conf["tile_size"]
    wsi_list_path = conf["wsi_list_path"]
    coords_dir = conf["coords_dir"]
    wsi_dir = conf["wsi_dir"]
    wsi_path_ext = conf["wsi_path_ext"]
    save_path = conf["save_path"]

    # tile_encoder = load_tile_encoder()
    tile_encoder = tile_encoder.cuda(gpu_id)

    df = pd.read_csv(wsi_list_path)
    wsi_names = df.slide_id

    for wsi_n in wsi_names:
        wsi_path = os.path.join(wsi_dir, wsi_n + wsi_path_ext)
        coords_path = os.path.join(coords_dir, wsi_n + ".h5")

        encode_wsi_tiles(
            resolution=resolution,
            batch_size=batch_size,
            tile_size=tile_size,
            coords_path=coords_path,
            wsi_path=wsi_path,
            save_path=save_path,
            tile_encoder=tile_encoder,
            gpu_id=gpu_id
        )


if __name__ == "__main__":
    mp.set_start_method('spawn')  # Use 'spawn' start method to avoid fork issues
    parser = argparse.ArgumentParser(description="parser function to encode WSI tiles")
    parser.add_argument("--conf", type=str, help="path to config file", default="tile_encoder_config.json")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        conf = json.load(f)

    tile_encoder = load_tile_encoder()
    world_size = torch.cuda.device_count()  # Get the number of available GPUs
    mp.spawn(run_inference_on_all_gpus, args=(conf,tile_encoder,), nprocs=world_size, join=True)

