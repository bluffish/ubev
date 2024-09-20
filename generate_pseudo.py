import diffusers
from diffusers import DiffusionPipeline
from tqdm import trange
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from rembg import remove

import numpy as np
import os
import argparse
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def crop_fit(image):
    data = np.array(image)
    alpha_channel = data[:, :, 3]
    non_empty_pixels = np.where(alpha_channel > 0)

    y_min, y_max = np.min(non_empty_pixels[0]), np.max(non_empty_pixels[0])
    x_min, x_max = np.min(non_empty_pixels[1]), np.max(non_empty_pixels[1])

    return image.crop((x_min, y_min, x_max + 1, y_max + 1))


def generate():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-0.9",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )

    pipe.set_progress_bar_config(disable=True)
    pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    pipe = pipe.to(args.gpus)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-0.9",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    refiner.set_progress_bar_config(disable=True)
    refiner.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    refiner.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    refiner = refiner.to(args.gpus)

    animals = ["bear", "horse", "deer", "moose", "cow", "donkey", "wolf", "fox", "elk", "lion"]

    for animal in animals:
        save_path = os.path.join(args.save_path, animal)
        os.makedirs(save_path, exist_ok=True)

        for i in trange(args.samples, desc=animal):
            prompt = f"a {animal} facing forwards. I want to see the whole body of the {animal}."
            # prompt = f"a {animal}. I want to see the entire {animal}"

            image = pipe(prompt, output_type="latent").images[0]
            image = refiner(prompt=prompt, image=image).images[0]
            image = remove(image)
            image = crop_fit(image)

            image.save(os.path.join(save_path, f"{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("save_path")
    parser.add_argument('-s', '--samples', required=False, type=int, default=200)
    parser.add_argument('-g', '--gpus', required=False, type=int, default=7)

    args = parser.parse_args()

    generate()
