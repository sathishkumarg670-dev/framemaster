import torch
import cv2
import json
import time
import torchreid
from pathlib import Path
from torchvision import transforms

MODEL = Path("models/reid/osnet_x1_0_market1501.pt")
INPUT_JSON = Path(
    "data/attributes/camera_01/objects/detected_object_crops.json"
)
OUT = Path(
    "data/reid/camera_01/object_embeddings.json"
)

model = torchreid.models.build_model(
    "osnet_x1_0",
    751,
    loss="softmax",
    pretrained=False,
    use_gpu=False
)

state = torch.load(
    MODEL,
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(state, strict=True)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

with open(INPUT_JSON, "r") as f:
    records = json.load(f)

data = []
batch = []
meta = []

start = time.time()

print("Total object crops:", len(records))

for i, record in enumerate(records, 1):

    path = Path(record["crop"])

    image = cv2.imread(str(path))

    if image is None:
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    batch.append(transform(image))
    meta.append(record)

    if len(batch) == 8 or i == len(records):

        with torch.no_grad():
            embeddings = model(torch.stack(batch))

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

        for record, embedding in zip(meta, embeddings):

            data.append({
                "object": record["object"],
                "frame": record["frame"],
                "confidence": record["confidence"],
                "bbox": record["bbox"],
                "crop": record["crop"],
                "embedding": embedding.tolist()
            })

        batch = []
        meta = []

        print(
            f"{i}/{len(records)} processed | "
            f"{time.time() - start:.1f}s"
        )

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUT, "w") as f:
    json.dump(data, f)

print()
print("DONE")
print("Embeddings:", len(data))
print("Embedding dimension:", len(data[0]["embedding"]))
print("Output:", OUT)