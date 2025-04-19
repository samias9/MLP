import numpy as np
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from scipy.interpolate import splprep, splev

def normalize_and_center(x, y, scale=1.0):
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    return np.stack((x * scale, y * scale), axis=1)

def star_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "star"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].to_numpy() + rng.normal(0, 0.15, n)
    y = df["y"].iloc[ix].to_numpy() + rng.normal(0, 0.15, n)

    x = (x / 54 - 1) * 4
    y = (y / 48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def dino_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def heart_dataset(n=8000, noise=0.05):
    rng = np.random.default_rng(42)
    t = rng.uniform(0, 2 * np.pi, n)
    x = 16 * np.sin(t) ** 3
    y = (13 * np.cos(t) -
         5 * np.cos(2 * t) -
         2 * np.cos(3 * t) -
         np.cos(4 * t))
    x += rng.normal(0, noise, n)
    y += rng.normal(0, noise, n)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    X = np.stack((x, y), axis=1)
    X *= 2
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def sun_dataset(n=8000, rays=12, noise_level=0.02):
    rng = np.random.default_rng(42)

    base_n = int(n * 0.5)
    t = np.linspace(0, 2 * np.pi, base_n)
    x = np.cos(t)
    y = np.sin(t)

    rays_n = n - base_n
    ray_t = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    ray_t = np.repeat(ray_t, rays_n // rays)

    ray_r = 1.2 + 0.3 * rng.random(len(ray_t))
    ray_x = ray_r * np.cos(ray_t)
    ray_y = ray_r * np.sin(ray_t)

    x = np.concatenate([x, ray_x])
    y = np.concatenate([y, ray_y])

    x += rng.normal(0, noise_level, len(x))
    y += rng.normal(0, noise_level, len(y))

    points = normalize_and_center(x, y, scale=2.5)
    return TensorDataset(torch.from_numpy(points.astype(np.float32)))

def get_dataset(name, n=8000):
    if name == "moons":
        from datasets import moons_dataset
        return moons_dataset(n)
    elif name == "dino":
        from datasets import dino_dataset
        return dino_dataset(n)
    elif name == "sun":
        from datasets import sun_dataset
        return sun_dataset(n)
    elif name == "heart":
        from datasets import heart_dataset
        return heart_dataset(n)
    elif name == "star":
        from datasets import star_dataset
        return star_dataset(n)
    elif name == "line":
        from datasets import line_dataset
        return line_dataset(n)
    elif name == "circle":
        from datasets import circle_dataset
        return circle_dataset(n)
    elif name == "spiral":
        from datasets import spiral_dataset
        return spiral_dataset(n)
    else:
        raise ValueError(f"Dataset inconnue: {name}")
