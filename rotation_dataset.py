import io
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

img_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")

conf = {
    "ru_printed": 65000,
    "TextOCR": 65000,
    "en_printed": 25000,
    "ru_wr": 15000,
    "en_wr": 8000,
}

split = (0.8, 0.1, 0.1)


def resize_wh(img, target_w=256, target_h=64, fill=(255, 255, 255)):
    w, h = img.size
    new_w = max(1, int(w * target_h / h))
    img = img.resize((new_w, target_h), Image.BILINEAR)
    c = Image.new("RGB", (target_w, target_h), fill)
    x = (target_w - new_w) // 2 if new_w <= target_w else -((new_w - target_w) // 2)
    c.paste(img, (x, 0))
    return c


def scan(path):
    return [os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs if f.lower().endswith(img_ext)]


class RotationDataset(Dataset):
    def __init__(self, root_dir="datasets", split="train", seed=27, img_width=256,img_height=64, domain_config=None, split_ratios=split, augment=None):
        assert split in ("train", "val", "test")
        self.split = split
        self.img_width =  img_width
        self.img_height = img_height
        rng = random.Random(seed)
        self.manifest = []
        self.domain_counts = {}

        for domain, target in (domain_config or conf).items():
            path = os.path.join(root_dir, domain)
            items = scan(path)
            if not items:
                print(f"{domain} отсутствует, пропускаю")
                continue
            rng.shuffle(items)
            n = min(target, len(items))
            items = items[:n]
            a, b = int(n * split_ratios[0]), int(n * (split_ratios[0] + split_ratios[1]))
            sel = {"train": items[:a], "val": items[a:b], "test": items[b:]}[split]
            self.domain_counts[domain] = len(sel)
            self.manifest.extend((domain, it) for it in sel)

        rng.shuffle(self.manifest)
        self.augment = augment or T.Compose([
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
            T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)], p=0.5),
            T.RandomApply([T.RandomRotation(degrees=8, fill=255)], p=0.5),
            T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=0.5)], p=0.2),
        ])
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.manifest)

    def _jpeg_recompress(self, img, quality_range=(30, 90)):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*quality_range))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __getitem__(self, idx):
        domain, path = self.manifest[idx]
        img = Image.open(path).convert("RGB")

        label = float(random.random() < 0.5)
        if label:
            img = img.rotate(180)

        img = resize_wh(img, self.img_width, self.img_height)

        if self.split == "train":
            img = self.augment(img)
            if random.random() < 0.3:
                img = self._jpeg_recompress(img)

        return {
            "image": self.to_tensor(img),
            "label": torch.tensor(label, dtype=torch.float32),
            "domain": domain,
        }

    def summary(self):
        print(f"[{self.split}] всего сэмплов: {len(self.manifest)}")
        for d, c in self.domain_counts.items():
            print(f"  {d}: {c}")