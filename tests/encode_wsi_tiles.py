import timm
from PIL import Image
from torchvision import transforms
import torch
import argparse
from cp_toolbox.image_processing.slide import Wsi
import cp_toolbox.deep_learning.torch.predict_wsi as predict_wsi_torch
import h5py
import torch.nn as nn
import json

class PrePostProcessModel(nn.Module):
    """
    inputs:
     - prepcocessing_layer_ transformes layer or a nn module
     - model: model
     - postprocessing_layer
    """
    def __init__(self, model, preprocessing_layer=None, postprocessing_layer=None, device=None):
        super(PrePostProcessModel, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = model.to(device)
        # if preprocessing_layer is not None:
        #     self.preprocessing_layer = preprocessing_layer.to(device)
        # if postprocessing_layer is not None:
        #     self.postprocessing_layer = postprocessing_layer.to(device)

        self.preprocessing_layer = preprocessing_layer
        self.postprocessing_layer = postprocessing_layer

    def forward(self, input):
        x = input.to(self.device)
        if self.preprocessing_layer is not None:
            x = self.preprocessing_layer(x)
        x = self.model(x)
        if self.postprocessing_layer is not None:
            x = self.postprocessing_layer(x)
        return {"out":x}


# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)

print(device)

# ### TileGenerator parameters
# resolution = 20
# batch_size = 10
# tile_size = 256
#
# filename = "7b3dc0e9-cbe0-479c-b9e2-7cafc40e2b65"
# features_path = f"/app/felipe/projects/CLAM/datasets/features/h5_files/{filename}.h5"
# wsi_path = f"/app/felipe/projects/CLAM/datasets/features/{filename}.tiff"


def encode_wsi_tiles(resolution, batch_size, tile_size, num_workers,
                     features_path, wsi_path):

    wsi = Wsi(wsi_path)

    with h5py.File(features_path, "r") as f:
        coords = f["coords"][()]  # returns as a numpy array
    f.close()

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model = PrePostProcessModel(model=tile_encoder, preprocessing_layer=transform, device=device)


    wsi_processor = predict_wsi_torch.WsiProcessor(
        wsi=wsi,
        model=model,
    #     model=model_2,
        resolution=resolution,
        tile_size=tile_size,
        target_size=tile_size,
        batch_size=batch_size,
        step_size=tile_size,
        filter_bgnd_tiles=False,
        preprocessing_layer=transform,
        cvt_color_code="rgb",  # default is bgr. the tf model was trained with bgr. OJO
        mode="classification",
        # device=device  # only needed for pytorch
        data_loader_kwargs = {"num_workers": num_workers, "pin_memory":True}
    )
    predict_kwargs = {"device":device}

    wsi_processor.coords = coords  # provide coords that were custom filtered

    results = wsi_processor.evaluate_wsi(device=device)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser function to encode wsi tiles")
    parser.add_argument("--conf", type=str, help="path to config file")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        conf = json.load(f)
    f.close()

    print(conf)

    resolution = conf["resolution"]
    batch_size = conf["batch_size"]
    tile_size = conf["tile_size"]
    num_workers = conf["num_workers"]
    features_path = conf["features_path"]
    wsi_path = conf["wsi_path"]

    encode_wsi_tiles(resolution, batch_size, tile_size, num_workers,
                     features_path, wsi_path)