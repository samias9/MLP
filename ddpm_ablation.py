# Ablation pour le taux d'apprentissage

from positional_embeddings import PositionalEmbedding
from schedulers import NoiseScheduler
from block import Block
from multiClassDataset import MultiClassDataset
import datasets
import argparse
import os
import imageio.v2 as imageio

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np


class ConditionalMLP(nn.Module):
    def __init__(self, hidden_size=128, hidden_layers=3, emb_size=128,
                 time_emb="sinusoidal", input_emb="sinusoidal", num_classes=1):
        super().__init__()
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.class_embedding = nn.Embedding(num_classes, emb_size)

        concat_size = len(self.time_mlp.layer) + len(self.input_mlp1.layer) + len(self.input_mlp2.layer) + emb_size

        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, class_labels=None):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        class_emb = self.class_embedding(class_labels) if class_labels is not None else torch.zeros(x.size(0), self.class_embedding.embedding_dim)
        x = torch.cat((x1_emb, x2_emb, t_emb, class_emb), dim=-1)
        return self.joint_mlp(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="conditional")
    parser.add_argument("--datasets", type=str, nargs='+', default=["human"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal")
    parser.add_argument("--input_embedding", type=str, default="sinusoidal")
    parser.add_argument("--save_images_step", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    args = parser.parse_args()

    learning_rates = [1e-4, 1e-3, 1e-2]  # valeurs d' Ablation pour le taux d'apprentissage

    for lr in learning_rates:
        experiment_name = f"{args.experiment_name}_lr{lr:.0e}"

        datasets_dict = {name: datasets.get_dataset(name) for name in args.datasets}
        multi_dataset = MultiClassDataset(datasets_dict)
        dataloader = DataLoader(multi_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        num_classes = len(args.datasets)

        model = ConditionalMLP(
            hidden_size=args.hidden_size,
            hidden_layers=args.hidden_layers,
            emb_size=args.embedding_size,
            time_emb=args.time_embedding,
            input_emb=args.input_embedding,
            num_classes=num_classes
        )

        noise_scheduler = NoiseScheduler(
            num_timesteps=args.num_timesteps,
            beta_schedule=args.beta_schedule
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        global_step = 0
        frames_by_class = {class_name: [] for class_name in datasets_dict.keys()}
        losses = []

        for epoch in range(args.num_epochs):
            model.train()
            progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} [lr={lr:.0e}]")
            for step, batch in enumerate(dataloader):
                points, class_labels = batch
                noise = torch.randn(points.shape)
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (points.shape[0],)).long()
                noisy = noise_scheduler.add_noise(points, noise, timesteps)
                noise_pred = model(noisy, timesteps, class_labels)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                global_step += 1
                progress_bar.set_postfix({"loss": loss.item(), "step": global_step})
                progress_bar.update(1)
            progress_bar.close()

            if epoch % args.save_images_step == 0 or epoch == args.num_epochs - 1:
                model.eval()
                for class_idx, class_name in enumerate(datasets_dict.keys()):
                    sample = torch.randn(args.eval_batch_size, 2)
                    class_labels = torch.full((args.eval_batch_size,), class_idx, dtype=torch.long)
                    timesteps = list(range(len(noise_scheduler)))[::-1]
                    for t in timesteps:
                        t_tensor = torch.full((args.eval_batch_size,), t, dtype=torch.long)
                        with torch.no_grad():
                            noise_pred_cond = model(sample, t_tensor, class_labels)
                            noise_pred_uncond = model(sample, t_tensor, None)
                            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                            sample = noise_scheduler.step(noise_pred, t, sample)
                    frames_by_class[class_name].append(sample.numpy())

        outdir = f"exps/{experiment_name}"
        os.makedirs(outdir, exist_ok=True)
        torch.save(model.state_dict(), f"{outdir}/model.pth")

        np.save(f"{outdir}/loss.npy", np.array(losses))

        for class_name, frames in frames_by_class.items():
            class_imgdir = f"{outdir}/images/{class_name}"
            os.makedirs(class_imgdir, exist_ok=True)
            frames_array = np.stack(frames)
            for i, frame in enumerate(frames_array):
                plt.figure(figsize=(10, 10))
                plt.scatter(frame[:, 0], frame[:, 1], alpha=0.7)
                plt.xlim(-6, 6)
                plt.ylim(-6, 6)
                plt.title(f"{class_name.capitalize()} - Epoch {i * args.save_images_step}")
                plt.savefig(f"{class_imgdir}/{i:04}.png")
                plt.close()
            gif_path = f"{outdir}/{class_name}_timelapse.gif"
            images = [imageio.imread(f"{class_imgdir}/{i:04}.png") for i in range(len(frames))]
            imageio.mimsave(gif_path, images, duration=0.3)

        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f"Training Loss - lr={lr:.0e}")
        plt.xlabel("Step")
        plt.ylabel("MSE Loss")
        plt.savefig(f"{outdir}/loss_curve.png")
        plt.close()
