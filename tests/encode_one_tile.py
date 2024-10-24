import timm
from PIL import Image
from torchvision import transforms
import torch
from huggingface_hub import login
login()

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)


transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

img_path = "01581x_25583y.png"
sample_input = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

tile_encoder.eval()
with torch.no_grad():
    output = tile_encoder(sample_input).squeeze()
    
print(output)
    
    
