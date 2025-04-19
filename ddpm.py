#python3 ddpm.py --dataset human

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
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np


class ConditionalMLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal", num_classes: int = 1):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        # Embedding de classe
        self.class_embedding = nn.Embedding(num_classes, emb_size)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer) + emb_size

        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, class_labels=None):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)

        # Si aucune étiquette n'est fournie, on utilise des zéros (génération non-conditionnelle)
        if class_labels is None:
            class_emb = torch.zeros(x.size(0), self.class_embedding.embedding_dim)
        else:
            class_emb = self.class_embedding(class_labels)

        x = torch.cat((x1_emb, x2_emb, t_emb, class_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="conditional")
    parser.add_argument("--datasets", type=str, nargs='+', default=["heart", "car", "human", "batman", "pikachu"],
                      help="List of datasets to train on")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                      help="Classifier-free guidance scale ")
    config = parser.parse_args()

   # Chargement des datasets spécifiés
    datasets_dict = {}
    for dataset_name in config.datasets:
        dataset = datasets.get_dataset(dataset_name)
        print(f"Dataset '{dataset_name}' type: {type(dataset)}")
        print(f"First element type: {type(dataset[0]) if hasattr(dataset, '__getitem__') else 'no __getitem__'}")
        datasets_dict[dataset_name] = dataset

    # Création du dataset multi-classe
    multi_dataset = MultiClassDataset(datasets_dict)
    dataloader = DataLoader(
        multi_dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    num_classes = len(config.datasets)

    model = ConditionalMLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding,
        num_classes=num_classes)

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    global_step = 0
    frames_by_class = {class_name: [] for class_name in datasets_dict.keys()}
    losses = []

    print("Training model...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            points, class_labels = batch

            # Échantillonner du bruit
            noise = torch.randn(points.shape)

            # Échantillonner des pas de temps
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (points.shape[0],)
            ).long()

            # Ajouter du bruit aux points
            noisy = noise_scheduler.add_noise(points, noise, timesteps)

            # Prédire le bruit (prédiction conditionnelle)
            noise_pred = model(noisy, timesteps, class_labels)

            # Calculer la perte
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            # Clippage des gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1

        progress_bar.close()

        # Générer des échantillons pour chaque classe
        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            model.eval()

            for class_idx, class_name in enumerate(datasets_dict.keys()):
                # Utiliser le guidage sans classificateur : générer des échantillons conditionnels et non-conditionnels
                samples_per_class = config.eval_batch_size

                # Initialiser avec du bruit aléatoire
                sample = torch.randn(samples_per_class, 2)

                # Préparer les étiquettes de classe : toutes appartiennent à la même classe
                class_labels = torch.full((samples_per_class,), class_idx, dtype=torch.long)

                # Lancer le processus de génération par diffusion
                timesteps = list(range(len(noise_scheduler)))[::-1]
                for i, t in enumerate(tqdm(timesteps, desc=f"Generating {class_name}")):
                    t_tensor = torch.full((samples_per_class,), t, dtype=torch.long)

                    with torch.no_grad():
                        # Prédiction conditionnelle
                        noise_pred_cond = model(sample, t_tensor, class_labels)

                        # Prédiction nonconditionnelle
                        noise_pred_uncond = model(sample, t_tensor, None)

                        # Appliquer le guidage sans classificateur
                        noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        # Mettre à jour l'échantillon avec l'étape du scheduler
                        sample = noise_scheduler.step(noise_pred, t, sample)

                # Sauvegarder les échantillons générés
                frames_by_class[class_name].append(sample.numpy())

    print("Sauvegarde du modele...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    with open(f"{outdir}/config.txt", "w") as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")

    print("Saving class-specific images...")
    for class_name, frames in frames_by_class.items():
        class_imgdir = f"{outdir}/images/{class_name}"
        os.makedirs(class_imgdir, exist_ok=True)

        frames_array = np.stack(frames)
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6

        for i, frame in enumerate(frames_array):
            plt.figure(figsize=(10, 10))
            plt.scatter(frame[:, 0], frame[:, 1], alpha=0.7)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.title(f"{class_name.capitalize()} - Epoch {i * config.save_images_step}")
            plt.savefig(f"{class_imgdir}/{i:04}.png")
            plt.close()

        # GIF pour la classe
        gif_path = f"{outdir}/{class_name}_timelapse.gif"
        images = [imageio.imread(f"{class_imgdir}/{i:04}.png") for i in range(len(frames))]
        imageio.mimsave(gif_path, images, duration=0.3)
        print(f"Saved {class_name} GIF to {gif_path}")

    plt.figure(figsize=(15, 10))
    for class_idx, class_name in enumerate(datasets_dict.keys()):
        final_samples = frames_by_class[class_name][-1]
        plt.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.7, label=class_name.capitalize())

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.title("Generated Samples for All Classes")
    plt.savefig(f"{outdir}/all_classes.png")
    plt.close()

    np.save(f"{outdir}/loss.npy", np.array(losses))

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.savefig(f"{outdir}/loss_curve.png")
    plt.close()

    print(f"Training complete! Results saved to {outdir}")
