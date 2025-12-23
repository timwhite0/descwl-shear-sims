"""
Unbatch image generation output into individual files.

Reads batch files from /nfs/turbo/lsa-regier/scratch/descwl/settingX/
and creates individual dataset_{idx}_size_1.pt files in settingX_individual/.

Usage:
    python unbatch_images.py --setting 1
    python unbatch_images.py --setting 1 --start-batch 1 --end-batch 5
"""
import argparse
import gc
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm


def get_batch_files(batch_dir):
    """Find and pair image/catalog batch files."""
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {batch_dir}")

    image_files = sorted(batch_path.glob("batch_*_images.pt"))
    catalog_files = sorted(batch_path.glob("batch_*_catalog.pt"))

    if len(image_files) != len(catalog_files):
        print(f"Warning: {len(image_files)} image files vs {len(catalog_files)} catalog files")

    # Extract batch numbers and pair files
    pairs = []
    for img_file in image_files:
        # Extract batch number from filename (e.g., batch_1_images.pt -> 1)
        batch_num = int(img_file.stem.split("_")[1])
        cat_file = batch_path / f"batch_{batch_num}_catalog.pt"
        if cat_file.exists():
            pairs.append((batch_num, img_file, cat_file))
        else:
            print(f"Warning: No catalog file for batch {batch_num}")

    pairs.sort(key=lambda x: x[0])  # Sort by batch number
    return pairs


def unbatch_setting(setting, base_input_dir, base_output_dir, start_batch=None, end_batch=None):
    """
    Unbatch all files for a given setting.

    Args:
        setting: Setting number (1-5)
        base_input_dir: Base directory containing settingX folders
        base_output_dir: Base directory for output (settingX_individual folders)
        start_batch: Optional starting batch number (1-indexed)
        end_batch: Optional ending batch number (inclusive)
    """
    input_dir = f"{base_input_dir}/setting{setting}"
    output_dir = f"{base_output_dir}/setting{setting}_individual"

    print(f"\n{'='*60}")
    print(f"Unbatching setting{setting}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Find batch files
    try:
        batch_pairs = get_batch_files(input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0

    if not batch_pairs:
        print("No batch files found!")
        return 0

    print(f"Found {len(batch_pairs)} batch file pairs")

    # Filter by batch range
    if start_batch is not None or end_batch is not None:
        start = start_batch or 1
        end = end_batch or max(p[0] for p in batch_pairs)
        batch_pairs = [(n, i, c) for n, i, c in batch_pairs if start <= n <= end]
        print(f"Processing batches {start} to {end} ({len(batch_pairs)} batches)")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process batches
    global_idx = 0
    total_saved = 0
    start_time = time.time()

    for batch_num, image_path, catalog_path in batch_pairs:
        print(f"\nProcessing batch {batch_num}...")

        # Load batch data
        images = torch.load(image_path, map_location="cpu")
        catalog = torch.load(catalog_path, map_location="cpu")

        batch_size = len(images)
        print(f"  Loaded {batch_size} images")

        # Process each image in the batch
        for local_idx in tqdm(range(batch_size), desc=f"Batch {batch_num}", unit="img"):
            # Extract single image data
            image = images[local_idx]
            n_sources = int(catalog["n_sources"][local_idx])
            locs = catalog["locs"][local_idx, :n_sources]  # Only valid locations
            shear_1 = float(catalog["shear_1"][local_idx])
            shear_2 = float(catalog["shear_2"][local_idx])

            # Create output dict matching expected format
            data = {
                "images": image,
                "tile_catalog": {
                    "locs": locs,
                    "n_sources": n_sources,
                    "shear_1": shear_1,
                    "shear_2": shear_2,
                },
            }

            # Save as list with single element (matching existing format)
            save_path = f"{output_dir}/dataset_{global_idx}_size_1.pt"
            torch.save([data], save_path)

            global_idx += 1
            total_saved += 1

        # Cleanup between batches
        del images, catalog
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Setting {setting} complete!")
    print(f"  Total images saved: {total_saved}")
    print(f"  Output files: dataset_0_size_1.pt to dataset_{global_idx-1}_size_1.pt")
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/total_saved:.3f}s per image)")
    print(f"{'='*60}")

    return total_saved


def main():
    parser = argparse.ArgumentParser(description="Unbatch image generation output")
    parser.add_argument("--setting", type=int, required=True, help="Setting number (1-5)")
    parser.add_argument("--start-batch", type=int, default=None, help="Starting batch number")
    parser.add_argument("--end-batch", type=int, default=None, help="Ending batch number")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/nfs/turbo/lsa-regier/scratch/descwl",
        help="Base input directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/nfs/turbo/lsa-regier/scratch/descwl",
        help="Base output directory",
    )
    args = parser.parse_args()

    print(f"Unbatch Images - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total = unbatch_setting(
        setting=args.setting,
        base_input_dir=args.input_dir,
        base_output_dir=args.output_dir,
        start_batch=args.start_batch,
        end_batch=args.end_batch,
    )

    print(f"\nDone! Saved {total} individual files.")


if __name__ == "__main__":
    main()
