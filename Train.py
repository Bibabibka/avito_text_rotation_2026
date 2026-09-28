import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from rotation_dataset import RotationDataset

data_dir = "datasets"
img_w, img_h = 256, 64
batch = 256
epochs = 25
lr = 2e-3
wd = 1e-4
ema_decay = 0.998
workers = 4
patience = 5
seed = 27
ckpt = "best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = device.type == "cuda"


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.bn2(self.conv2(f.relu(self.bn1(self.conv1(x)), inplace=True)))
        return f.relu(out + x, inplace=True)


def down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SimpleCNNRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = down(3, 16)
        self.stage1 = nn.Sequential(down(16, 32), ResBlock(32))
        self.stage2 = nn.Sequential(down(32, 64), ResBlock(64))
        self.stage3 = nn.Sequential(down(64, 96), ResBlock(96))
        self.pool = nn.AdaptiveAvgPool2d((4, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(96 * 4, 64), nn.ReLU(inplace=True),
                                        nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        x = self.pool(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        return self.classifier(x).squeeze(1)


def make_loader(split):
    ds = RotationDataset(root_dir=data_dir, split=split, seed=seed, img_width=img_w, img_height=img_h)
    ds.summary()
    is_train = split == "train"
    return DataLoader(ds, batch_size=batch, shuffle=is_train, num_workers=workers, pin_memory=True,
                      drop_last=is_train, persistent_workers=workers > 0)


@torch.no_grad()
def collect(model, loader):
    model.eval()
    logits, labels, domains = [], [], []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        logits.append((0.5 * (model(x) - model(torch.flip(x, dims=[2, 3])))).float().cpu())
        labels.append(b["label"].float())
        domains.extend(b["domain"])
    return torch.sigmoid(torch.cat(logits)).numpy(), torch.cat(labels).numpy(), np.array(domains)


def score(probs, labels):
    return 1 - np.mean((probs - labels) ** 2), np.mean((probs > 0.5) == labels)


def evaluate(model, loader):
    probs, labels, _ = collect(model, loader)
    return score(probs, labels)


def evaluate_by_domain(model, loader):
    probs, labels, domains = collect(model, loader)
    for d in sorted(set(domains)):
        m = domains == d
        s, acc = score(probs[m], labels[m])
        print(f"  {d:12s} n={m.sum():6d}  1-brier={s:.4f}  acc={acc:.4f}")


def main():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_loader, val_loader = make_loader("train"), make_loader("val")

    model = SimpleCNNRes().to(device).to(memory_format=torch.channels_last)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.15, anneal_strategy="cos", div_factor=10, final_div_factor=100)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = nn.BCEWithLogitsLoss()

    best, bad = -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for b in train_loader:
            x = b["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = b["label"].to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = crit(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_parameters(model)
            total += loss.item() * x.size(0)

        s, acc = evaluate(ema.module, val_loader)
        print(f"{epoch}/{epochs} _ train_loss={total / (len(train_loader) * batch):.4f}"
              f"_ val_1-brier(ema+tta)={s:.4f} _ val_acc={acc:.4f}")

        if s > best:
            best, bad = s, 0
            torch.save(ema.module.state_dict(), ckpt)
            print(f"сохранена лучшая модель в {ckpt}")
        else:
            bad += 1
            if bad >= patience:
                print(f" нет улучшения ")
                break

    print(f"лучший val 1-brier: {best:.4f}")

    model = SimpleCNNRes().to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    evaluate_by_domain(model, val_loader)


if __name__ == "__main__":
    main()