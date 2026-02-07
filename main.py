from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

import torch
import torch.nn.functional as F
from torchvision import transforms

from .model import BetterCNN  # model.py から import

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

# SvelteKit dev server から叩くなら CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNISTと同じ前処理（学習時と揃える）
tf = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # [0,1], shape=[1,28,28]
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# ----------------------------
# Load model at startup
# ----------------------------
model = BetterCNN(num_classes=10).to(device)
state = torch.load("mnist_cnn.pt", map_location=device)
model.load_state_dict(state)
model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@torch.no_grad()
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    b = await file.read()

    # PNG/JPG bytes -> PIL
    img = Image.open(io.BytesIO(b))

    # 透過PNGでも壊れないように黒背景に合成
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")

    # preprocess -> [1,1,28,28]
    x = tf(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    prob = float(probs[pred].item())

    return {
        "pred": pred,
        "prob": prob,
        "probs": probs.cpu().tolist(),
    }
