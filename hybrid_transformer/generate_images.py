import argparse
import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from transformers import ImageGPTFeatureExtractor, ImageGPTForCausalImageModeling


def generate_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_images(feature_extractor, model, batch_size, num_iters, path_to_generated_images, device):
    index = list(generate_chunks([i for i in range(num_iters)], batch_size))
    images = np.random.rand(num_iters, 3, 32, 32)
    for i in tqdm(range(len(index))):
        images[index[i]] = generate_images_batch(feature_extractor, model, len(index[i]), device)
        np.savez(path_to_generated_images, images[:index[i][-1]])
    np.savez(path_to_generated_images, images)
    return images


@torch.no_grad()
def generate_images_batch(feature_extractor, model, batch_size, device):
    context = torch.full((batch_size, 1), model.config.vocab_size - 1)  # initialize with SOS token (with ID 512)
    context = torch.Tensor(context).to(device)

    output = model.generate(input_ids=context, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True,
                            top_k=40)

    clusters = feature_extractor.clusters
    n_px = feature_extractor.size

    samples = output[:, 1:].cpu().detach().numpy()
    samples_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [32, 32, 3]).astype(np.uint8) for s in samples]
    samples_img = [img.transpose(2, 0, 1) for img in samples_img]
    return samples_img

def main():
    SIZE = 'medium'
    PRETRAINED_MODEL = f'openai/imagegpt-{SIZE}'
    NUM_IMAGES_TO_GENERATE = 10000
    BATCH_SIZE = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = ImageGPTFeatureExtractor.from_pretrained(PRETRAINED_MODEL)
    print("Loading model...")
    model = ImageGPTForCausalImageModeling.from_pretrained(PRETRAINED_MODEL)
    model.eval()
    model.to(device)

    generate_images(
        feature_extractor=feature_extractor,
        model=model,
        batch_size=BATCH_SIZE,
        num_iters=NUM_IMAGES_TO_GENERATE,
        device=device,
        path_to_generated_images=f'sampled-{SIZE}.npz')

if __name__ == "__main__":
    main()
