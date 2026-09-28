import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from Train import SimpleCNNRes


def letterbox_resize(img, target_w=256, target_h=64, fill=(255, 255, 255)):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    canvas.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


class TestDataset(Dataset):
    def __init__(self, df, test_dir):
        self.df = df
        self.file_map = {}
        for root, _, files in os.walk(test_dir):
            for file in files:
                self.file_map[os.path.splitext(file)[0]] = os.path.join(root, file)
        self.transform = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = str(self.df.iloc[idx]["image_id"]).strip()
        path = self.file_map.get(img_id)
        if path is None:
            return torch.zeros(3, 64, 256), idx
        img = letterbox_resize(Image.open(path))
        return self.transform(img), idx


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNNRes().to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    csv_path = "test/sample_submission.csv" if os.path.exists("test/sample_submission.csv") else "sample_submission.csv"
    df = pd.read_csv(csv_path)
    loader = DataLoader(TestDataset(df, "test"), batch_size=256, shuffle=False, num_workers=2)

    preds = [0.5] * len(df)
    with torch.no_grad():
        for imgs, indices in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            logits_flipped = model(torch.flip(imgs, dims=[2, 3]))
            logits_tta = 0.5 * (logits - logits_flipped)

            probs = torch.sigmoid(logits_tta).cpu().numpy()
            for i, p in zip(indices.numpy(), probs):
                preds[i] = float(p)

    df["p_180"] = preds
    df[["image_id", "p_180"]].to_csv("submission.csv", index=False)
    print("submission.csv готов")


if __name__ == "__main__":
    run_inference()