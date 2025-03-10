import argparse
import datetime
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble inference tool")
    parser.add_argument("--config", help="inference config file path")
    parser.add_argument(
        "--method",
        choices=["majority_vote", "weighted_majority_vote", "average", "intersection"],
        default="majority_vote",
        help="select ensemble method [majority_vote, weighted_majority_vote, average, intersection]",
    )
    return parser.parse_args()


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask, 0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def decode_rle_to_mask(rle, height=2048, width=2048):
    """
    Decodes RLE string to a binary mask.
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(height, width)


def majority_vote_ensemble(masks):
    """
    Majority vote ensemble method.
    """
    masks = np.stack(masks, axis=0)
    return (np.mean(masks, axis=0) >= 0.5).astype(np.uint8)


def weighted_majority_vote_ensemble(masks, weights):
    """
    Weighted Majority vote ensemble method.
    
    masks: list of numpy arrays (binary masks from different models)
    weights: list of weights corresponding to each model
    """
    masks = np.stack(masks, axis=0)  # Shape: (num_models, height, width)
    weights = np.array(weights).reshape(-1, 1, 1)  # Shape: (num_models, 1, 1)
    weighted_sum = np.sum(masks * weights, axis=0)
    threshold = np.sum(weights) / 2
    return (weighted_sum >= threshold).astype(np.uint8)


def average_ensemble(masks, threshold=0.5):
    """
    Average-based ensemble method.
    """
    masks = np.stack(masks, axis=0)
    return (np.mean(masks, axis=0) >= threshold).astype(np.uint8)


def intersection_ensemble(masks):
    """
    Intersection-based ensemble method.
    """
    masks = np.stack(masks, axis=0)
    return np.all(masks, axis=0).astype(np.uint8)


def ensemble_and_save(
    csv_paths, save_path, weights_config, height=2048, width=2048, method="majority_vote"
):
    """
    Reads CSV files, performs ensemble, and saves results to a file.
    
    weights_config: dictionary mapping class_id to list of weights per model
    """
    ensemble_data = {}
    num_models = len(csv_paths)
    
    # Collect data from all CSVs
    for csv_idx, csv_path in enumerate(tqdm(csv_paths, desc="Processing CSV files")):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            image_name = row["image_name"]
            class_id = row["class"]
            rle = row["rle"]

            # Skip missing RLEs
            if pd.isna(rle):
                print(f"Missing RLE: {image_name}, {class_id}")
                continue

            mask = decode_rle_to_mask(rle, height=height, width=width)
            if image_name not in ensemble_data:
                ensemble_data[image_name] = {}
            if class_id not in ensemble_data[image_name]:
                ensemble_data[image_name][class_id] = [None] * num_models
            ensemble_data[image_name][class_id][csv_idx] = mask

    # Perform ensemble and save results line by line
    with open(save_path, "w") as f:
        f.write("image_name,class,rle\n")  # Write CSV header

        for image_name, classes in tqdm(
            ensemble_data.items(), desc="Ensembling and saving results"
        ):
            for class_id, masks in classes.items():
                # Remove None masks (in case some models did not predict this class)
                masks = [m for m in masks if m is not None]
                if not masks:
                    print(f"No masks for: {image_name}, {class_id}")
                    continue

                if method == "majority_vote":
                    final_mask = majority_vote_ensemble(masks)
                elif method == "weighted_majority_vote":
                    # Retrieve weights for this class
                    class_weights = weights_config.get(class_id, [1] * len(masks))
                    if len(class_weights) != len(masks):
                        raise ValueError(f"Number of weights for class {class_id} does not match number of models.")
                    final_mask = weighted_majority_vote_ensemble(masks, class_weights)
                elif method == "average":
                    final_mask = average_ensemble(masks)
                elif method == "intersection":
                    final_mask = intersection_ensemble(masks)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Encode mask to RLE
                rle = encode_mask_to_rle(final_mask)

                # Save one line to file
                f.write(f"{image_name},{class_id},{rle}\n")


def main(args):
    cfg = OmegaConf.load(args.config)
    csv_paths = list(cfg.csv_paths)
    save_filename = str(cfg.save_filename)
    method = args.method

    # Load weights from config
    if method == "weighted_majority_vote":
        weights_config = cfg.weights
        if not weights_config:
            raise ValueError("Weights configuration is required for weighted_majority_vote method.")
    else:
        weights_config = None

    save_dir = "./ensemble_results"
    os.makedirs(save_dir, exist_ok=True)

    # Define output file path
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/{save_filename}_{method}_{current_time}.csv"

    # Perform ensemble and save results
    ensemble_and_save(
        csv_paths, save_path, weights_config=weights_config, method=method
    )

    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
