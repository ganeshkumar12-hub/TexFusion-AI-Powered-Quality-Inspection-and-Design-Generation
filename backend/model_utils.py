import os
import io
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torchvision.utils as utils
import cv2

# ---------------------------------------------------
# GENERATOR NETWORK (same architecture you trained)
# ---------------------------------------------------

class Generator(nn.Module):
    def __init__(self, nz, label_emb_dim, ngf, nc, num_classes, image_size=64):
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
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        le = self.label_emb(labels)
        x = torch.cat([z, le], dim=1)
        x = self.fc(x)
        return self.net(x)


# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1) / 2.0
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------
# LOAD GENERATOR CHECKPOINT
# ---------------------------------------------------

def load_generator(ckpt_path, num_classes):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    netG = Generator(
        nz=100,
        label_emb_dim=50,
        ngf=64,
        nc=3,
        num_classes=num_classes
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if "netG_state_dict" in ckpt:
        netG.load_state_dict(ckpt["netG_state_dict"])
    elif "model_state_dict" in ckpt:
        netG.load_state_dict(ckpt["model_state_dict"])
    else:
        netG.load_state_dict(ckpt)

    netG.eval()
    return netG


# ---------------------------------------------------
# SIMPLE GENERATION (core GAN output)
# ---------------------------------------------------

def generate_base_design(netG, label_idx, seed=0):
    device = next(netG.parameters()).device

    torch.manual_seed(seed)
    z = torch.randn(1, 100, device=device)
    labels = torch.tensor([label_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        out = netG(z, labels)[0]

    img = tensor_to_pil(out)
    img = img.resize((256, 256), Image.LANCZOS)
    return img


# ---------------------------------------------------
# RECOLOR
# ---------------------------------------------------

def recolor(img, hue=0, sat=1.0, val=1.0):
    rgb = np.array(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    hsv[..., 0] = (hsv[..., 0] + hue) % 180
    hsv[..., 1] *= float(sat)
    hsv[..., 2] *= float(val)

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    recolored = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(recolored)


# ---------------------------------------------------
# ENHANCEMENT
# ---------------------------------------------------

def enhance(img):
    arr = np.array(img)
    blur = cv2.GaussianBlur(arr, (3, 3), 0)
    sharp = cv2.addWeighted(arr, 1.5, blur, -0.5, 0)
    return Image.fromarray(sharp)


# ---------------------------------------------------
# GRID + WEAVE (simple final touches)
# ---------------------------------------------------

def tile(img, tiles=2):
    w, h = img.size
    grid = Image.new("RGB", (w * tiles, h * tiles))
    for i in range(tiles):
        for j in range(tiles):
            grid.paste(img, (i * w, j * h))
    return grid


def weave(img):
    arr = np.array(img)
    overlay = np.zeros_like(arr)
    overlay[:, ::6] = 230
    overlay[::6, :] = 230
    blended = cv2.addWeighted(arr, 1.0, overlay, 0.15, 0)
    return Image.fromarray(blended)


# ---------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------

def full_pipeline(netG, class_idx, seed=0, hue=0, sat=1.0, val=1.0):
    img = generate_base_design(netG, class_idx, seed)
    img = recolor(img, hue, sat, val)
    img = enhance(img)
    img = tile(img, 2)
    img = weave(img)
    return img
