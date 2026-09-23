from pathlib import Path
import torch
import torchreid
from PIL import Image

# ==========================================
# VISIONFORGE AI - M5.1
# Re-ID Model Test
# ==========================================

PROJECT_ROOT = Path.cwd()

CROP_DIR = PROJECT_ROOT / "data" / "crops" / "camera_01"

# Pick the first available crop
crop_files = sorted(CROP_DIR.glob("*.jpg"))

if not crop_files:
    raise FileNotFoundError("No person crops found.")

TEST_IMAGE = crop_files[0]

print("===================================")
print("       VISIONFORGE AI")
print("       RE-ID MODEL TEST")
print("===================================")
print()
print(f"Test image: {TEST_IMAGE}")
print()

# ------------------------------------------
# Build OSNet model
# ------------------------------------------

print("Loading OSNet...")

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

# CPU
device = torch.device("cpu")
model = model.to(device)

# Evaluation mode
model.eval()

print("OSNet loaded successfully!")
print()

# ------------------------------------------
# Load image
# ------------------------------------------

image = Image.open(TEST_IMAGE).convert("RGB")

# Torchreid image preprocessing
transform = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    is_train=False
)[1]

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(device)

# ------------------------------------------
# Generate embedding
# ------------------------------------------

print("Generating embedding...")

with torch.no_grad():
    embedding = model(image_tensor)

# Flatten if necessary
embedding = embedding.view(embedding.size(0), -1)

# Normalize
embedding = torch.nn.functional.normalize(
    embedding,
    p=2,
    dim=1
)

print()
print("===================================")
print("Re-ID test completed!")
print("===================================")
print(f"Embedding shape : {tuple(embedding.shape)}")
print(f"Embedding size  : {embedding.shape[1]}")
print()
print("First 10 values:")
print(embedding[0][:10])
print()
print("CPU Re-ID model is working!")