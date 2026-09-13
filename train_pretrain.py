"""
Preentrenamiento del VAE sobre ZINC250k.
==========================================
Correr esto en GPU (Colab / cluster), no en un entorno de 1 CPU.
En este entorno solo se valida que el codigo corre correctamente (smoke test).

Uso (GPU real):
    python data_pipeline.py                       # dataset completo
    python train_pretrain.py --epochs 50 --batch_size 128 --device cuda

Uso (smoke test en CPU, este entorno):
    python data_pipeline.py --n_sample 500 --out_dir processed_smoke
    python train_pretrain.py --data_dir processed_smoke --epochs 2 --batch_size 32 --device cpu
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from vae_model import SelfiesVAE, vae_loss, kl_anneal_schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="processed")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent_dim", type=int, default=56)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "vocab.json")) as f:
        meta = json.load(f)
    vocab = meta["vocab"]
    pad_idx = vocab["[pad]"]

    train_data = torch.load(os.path.join(args.data_dir, "train.pt"))
    val_data = torch.load(os.path.join(args.data_dir, "val.pt"))
    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=args.batch_size)

    model = SelfiesVAE(vocab_size=len(vocab), hidden_size=args.hidden_size,
                        latent_dim=args.latent_dim, pad_idx=pad_idx).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_steps = args.epochs * len(train_loader)
    step = 0
    print(f"[train_pretrain] device={args.device} | vocab={len(vocab)} | "
          f"train={len(train_data)} | val={len(val_data)} | total_steps={total_steps}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_recon, running_kl = 0.0, 0.0
        for (batch,) in train_loader:
            batch = batch.to(args.device)
            beta = kl_anneal_schedule(step, total_steps)

            logits, mu, logvar, _ = model(batch)
            loss, recon, kl = vae_loss(logits, batch, mu, logvar, pad_idx=pad_idx, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_recon += recon
            running_kl += kl
            step += 1

        n_batches = len(train_loader)
        print(f"[epoch {epoch+1}/{args.epochs}] recon={running_recon/n_batches:.4f} "
              f"kl={running_kl/n_batches:.4f} beta={beta:.3f} "
              f"({time.time()-t0:.1f}s)")

        ckpt_path = os.path.join(args.ckpt_dir, f"vae_epoch{epoch+1}.pt")
        torch.save({"model_state": model.state_dict(), "vocab": vocab,
                    "latent_dim": args.latent_dim, "hidden_size": args.hidden_size}, ckpt_path)

    print(f"[train_pretrain] Entrenamiento terminado. Checkpoints en {args.ckpt_dir}/")


if __name__ == "__main__":
    main()
