print("APP FILE STARTED")

# ==========================================
# DISABLE ALL TENSORFLOW LOGGING (MUST BE FIRST)
# ==========================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

# ==========================================
# IMPORTS
# ==========================================
import io
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

import keras   # KERAS 3
from keras.applications.efficientnet import preprocess_input   # ⭐ IMPORTANT


# ==========================================
# SETTINGS
# ==========================================
MODEL_PATH = "models/modeltexgen.pth"
QUALITY_MODEL_PATH = "models/best_cloth_model.keras"
PATTERN_MODEL_PATH = "models/efficientnetB3_best.keras"
QUALITY_LABELS_PATH = "class_labels.txt"

PATTERN_LABELS = [
    "argyle", "camouflage", "checked", "dot", "floral",
    "geometric", "gradient", "graphic", "houndstooth",
    "leopard", "lettering", "muji", "paisley", "snake_skin",
    "snow_flake", "stripe", "tropical", "zebra", "zigzag"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORIES = [
    "abstract", "animal_print", "checks", "floral", "geometric",
    "houndstooth", "plaid", "plain", "polka_dots", "stripes",
    "waves", "zigzag"
]


# ==========================================
# FLASK
# ==========================================
app = Flask(__name__)
CORS(app)


# ==========================================
# GAN MODEL (Design Generation)
# ==========================================
nz = 100
ngf = 64
nc = 3
label_emb_dim = 50
num_classes = 12


class Generator(nn.Module):
    def __init__(self, nz, label_emb_dim, ngf, nc, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        input_dim = nz + label_emb_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True)
        )

        self.net = nn.Sequential(
            nn.Unflatten(1, (ngf * 8, 4, 4)),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        le = self.label_emb(labels)
        x = torch.cat([z, le], dim=1)
        x = self.fc(x)
        return self.net(x)


print("Loading GAN model...")
netG = Generator(nz, label_emb_dim, ngf, nc, num_classes).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
netG.load_state_dict(state["netG_state_dict"])
netG.eval()
print("GAN model loaded!")


# ==========================================
# QUALITY INSPECTION MODEL
# ==========================================
print("Loading Quality Inspection model...")
quality_model = keras.models.load_model(QUALITY_MODEL_PATH, compile=False)

with open(QUALITY_LABELS_PATH, "r") as f:
    QUALITY_LABELS = [line.strip() for line in f.readlines()]

print("Quality Inspection model loaded!")


# ==========================================
# PATTERN RECOGNITION MODEL
# ==========================================
print("Loading Pattern Recognition model...")
pattern_model = keras.models.load_model(PATTERN_MODEL_PATH, compile=False)
print("Pattern Recognition model loaded!")


# ==========================================
# HELPERS
# ==========================================
def tensor_to_pil(t):
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1) / 2
    arr = (t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def preprocess_quality_image(pil_img):
    img = pil_img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)


# ⭐ EXACT TRAINING PIPELINE
def preprocess_pattern_image(pil_img):
    img = pil_img.resize((300, 300))
    img = np.array(img).astype("float32")
    img = preprocess_input(img)  # EXACT SAME AS COLAB
    return np.expand_dims(img, axis=0)


# ------------------------------------------------
# Motif Overlay + Enhancement
# ------------------------------------------------

def overlay_motif(base, motif, mode="none", scale=0.2, spacing=30):
    base = base.convert("RGBA")
    motif = motif.convert("RGBA")

    ow, oh = base.size
    mw = int(ow * scale)
    motif = motif.resize((mw, mw), Image.LANCZOS)

    out = base.copy()

    if mode == "single":
        x = (ow - mw) // 2
        y = (oh - mw) // 2
        out.alpha_composite(motif, dest=(x, y))

    elif mode == "spread":
        for y in range(0, oh, mw + spacing):
            for x in range(0, ow, mw + spacing):
                out.alpha_composite(motif, dest=(x, y))

    return out


def enhance_design(image, mode="sharpen"):
    img = np.array(image)

    if mode == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    elif mode == "smooth":
        img = cv2.GaussianBlur(img, (5, 5), 0)

    elif mode == "watercolor":
        img = cv2.bilateralFilter(img, 9, 75, 75)

    return Image.fromarray(img)


# ==========================================
# QUALITY INSPECTION API
# ==========================================
@app.route("/api/quality-inspection", methods=["POST"])
def quality_inspection():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        pil_img = Image.open(file.stream).convert("RGB")

        img_array = preprocess_quality_image(pil_img)
        pred = quality_model.predict(img_array, verbose=0)[0]

        class_index = int(np.argmax(pred))
        confidence = float(np.max(pred) * 100)

        top3_idx = pred.argsort()[-3:][::-1]
        top3 = [
            {"label": QUALITY_LABELS[i], "score": round(float(pred[i] * 100), 2)}
            for i in top3_idx
        ]

        return jsonify({
            "prediction": QUALITY_LABELS[class_index],
            "confidence": round(confidence, 2),
            "top3": top3
        })

    except Exception as e:
        print("QUALITY ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ==========================================
# PATTERN RECOGNITION API
# ==========================================
@app.route("/api/pattern-recognition", methods=["POST"])
def pattern_recognition():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        pil_img = Image.open(file.stream).convert("RGB")

        img_array = preprocess_pattern_image(pil_img)
        pred = pattern_model.predict(img_array, verbose=0)[0]

        class_index = int(np.argmax(pred))
        confidence = float(np.max(pred) * 100)

        return jsonify({
            "pattern": PATTERN_LABELS[class_index],
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("PATTERN ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ==========================================
# DESIGN GENERATION API (WITH MOTIF)
# ==========================================
@app.route("/api/generate-design", methods=["POST"])
def generate_design():
    try:
        category = request.form.get("category", "plain")
        hue = float(request.form.get("hue", 0))
        sat = float(request.form.get("sat", 1.0))
        val = float(request.form.get("val", 1.0))

        # motif params
        motif_file = request.files.get("motif")
        motifMode = request.form.get("motifMode", "none")
        motifScale = float(request.form.get("motifScale", 0.2))
        motifSpacing = int(request.form.get("motifSpacing", 30))
        enhanceMode = request.form.get("enhanceMode", "sharpen")

        class_idx = CATEGORIES.index(category)
        z = torch.randn(1, nz, device=DEVICE)
        labels = torch.tensor([class_idx], device=DEVICE)

        with torch.no_grad():
            fake = netG(z, labels)[0]

        img = tensor_to_pil(fake).resize((512, 512))

        # HSV recolor
        rgb = np.array(img)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        final_img = Image.fromarray(rgb)

        # MOTIF OVERLAY
        if motif_file and motifMode != "none":
            motif_img = Image.open(motif_file.stream).convert("RGBA")
            final_img = overlay_motif(final_img, motif_img, motifMode, motifScale, motifSpacing)

        # enhancement
        final_img = enhance_design(final_img, enhanceMode)

        # send output
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        buf.seek(0)

        resp = make_response(buf.getvalue())
        resp.headers.set("Content-Type", "image/png")
        return resp

    except Exception as e:
        print("DESIGN ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5001)
