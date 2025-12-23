"""
Consolidated image generation script for weak lensing simulations.
Generates synthetic astronomical images with controlled shear effects.

Usage:
    python generate_images.py
    python generate_images.py --config path/to/config.yaml

Config is read from sim_config.yaml in the same directory by default.
"""
import os
import gc
import time
import tempfile
import shutil
import argparse
import yaml
from datetime import datetime

import torch
import numpy as np
from numpy.random import SeedSequence
import galsim
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing as mp

from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.stars import StarCatalog, make_star_catalog
from descwl_shear_sims.sim import make_sim, get_se_dim
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.layout.layout import Layout


# =============================================================================
# Utility Functions
# =============================================================================

def cleanup_memory():
    """Force garbage collection to free memory."""
    gc.collect()


def safe_save_tensor(tensor, filepath, max_retries=3):
    """
    Save tensor to disk with retry logic and atomic write.

    Uses temp file + move for atomic operation to prevent corruption.
    """
    print(f"Saving to {os.path.basename(filepath)}...")

    if hasattr(tensor, 'detach'):
        clean_tensor = tensor.detach().cpu().contiguous()
    else:
        clean_tensor = tensor

    target_dir = os.path.dirname(filepath)
    target_filename = os.path.basename(filepath)

    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(
                suffix='.tmp',
                prefix=f'{target_filename}.',
                dir=target_dir,
                delete=False
            ) as temp_file:
                temp_path = temp_file.name

            torch.save(clean_tensor, temp_path, _use_new_zipfile_serialization=False)

            temp_size = os.path.getsize(temp_path)
            if temp_size < 100:
                raise RuntimeError(f"File too small: {temp_size} bytes")

            shutil.move(temp_path, filepath)

            if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
                size_mb = os.path.getsize(filepath) / (1024**2)
                print(f"  Saved successfully ({size_mb:.1f} MB)")
                return True
            else:
                raise RuntimeError("Final file missing or corrupted")

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")

            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

            if attempt < max_retries - 1:
                time.sleep(1)

    print(f"All save attempts failed for {filepath}")
    return False


# =============================================================================
# PSF Functions
# =============================================================================

def get_psf_param(psf_obj, return_image=True, psf_size=64, center_pos=None):
    """
    Extract PSF parameters and optionally the PSF image array.

    Parameters
    ----------
    psf_obj : PSF object
        The PSF object from simulation (variable PSF, Gaussian, or Moffat)
    return_image : bool
        Whether to include the actual PSF image array
    psf_size : int
        Size of the PSF image to generate (psf_size x psf_size)
    center_pos : tuple or None
        Position to evaluate PSF at. If None, uses center.

    Returns
    -------
    dict : PSF statistics and optionally the PSF image
    """
    if hasattr(psf_obj, '_im_cen') and hasattr(psf_obj, '_tot_width'):
        # Variable PSF
        if center_pos is None:
            center_pos = (psf_obj._im_cen, psf_obj._im_cen)

        x_pixels = np.arange(psf_obj._tot_width)
        y_pixels = np.arange(psf_obj._tot_width)
        X, Y = np.meshgrid(x_pixels, y_pixels)

        X_coord = (X - psf_obj._im_cen) * psf_obj._scale
        Y_coord = (Y - psf_obj._im_cen) * psf_obj._scale

        g1_sampled, g2_sampled, mu_sampled = psf_obj._get_lensing((X_coord, Y_coord))

        psf_stats = {
            'g1_mean': float(np.mean(g1_sampled)),
            'g1_var': float(np.var(g1_sampled)),
            'g2_mean': float(np.mean(g2_sampled)),
            'g2_var': float(np.var(g2_sampled)),
            'mu_mean': float(np.mean(mu_sampled)),
            'mu_var': float(np.var(mu_sampled)),
            'psf_type': 'variable'
        }

        if return_image:
            galsim_pos = galsim.PositionD(center_pos[0], center_pos[1])
            psf_galsim_obj = psf_obj.getPSF(galsim_pos)

            psf_image = psf_galsim_obj.drawImage(
                nx=psf_size,
                ny=psf_size,
                scale=psf_obj._scale,
                method='auto'
            ).array

            psf_image = psf_image / np.sum(psf_image)
            psf_stats['psf_image'] = psf_image.astype(np.float32)
            psf_stats['psf_scale'] = psf_obj._scale
    else:
        # Fixed PSF
        if center_pos is None:
            center_pos = (psf_size / 2.0, psf_size / 2.0)

        psf_stats = {
            'g1_mean': 0.0,
            'g1_var': 0.0,
            'g2_mean': 0.0,
            'g2_var': 0.0,
            'mu_mean': 1.0,
            'mu_var': 0.0,
            'psf_type': 'fixed'
        }

        if return_image:
            pixel_scale = 0.2

            psf_image = psf_obj.drawImage(
                nx=psf_size,
                ny=psf_size,
                scale=pixel_scale,
                method='auto'
            ).array

            psf_image = psf_image / np.sum(psf_image)
            psf_stats['psf_image'] = psf_image.astype(np.float32)
            psf_stats['psf_scale'] = pixel_scale

    if return_image:
        psf_stats.update({
            'psf_size': psf_size,
            'psf_center_pos': center_pos
        })

    return psf_stats


def create_psf(config, se_dim, rng_seed):
    """Create PSF object based on config."""
    rng = np.random.RandomState(rng_seed)

    if config.get('variable_psf', False):
        return make_ps_psf(rng=rng, dim=se_dim, variation_factor=config['variation_factor'])
    else:
        return make_fixed_psf(psf_type=config['psf_type'], psf_fwhm=config.get('psf_fwhm', 0.8))


def get_optional_psf_param(psf, config):
    """Get PSF parameters if configured, otherwise return None."""
    if config.get('save_psf_param', False):
        return get_psf_param(psf, return_image=True, psf_size=config.get('psf_size', 64))
    return None


# =============================================================================
# Star Catalog Functions
# =============================================================================

def filter_bright_stars_production(star_catalog, mag_threshold=18, band='r', verbose=True):
    """
    Filter out bright stars for performance improvement.

    Removing bright stars (mag < threshold) prevents expensive draw_bright_star() calls.
    At mag_threshold=18, this provides ~91% speedup.

    Parameters
    ----------
    star_catalog : StarCatalog
        Original star catalog
    mag_threshold : float
        Stars brighter than this (lower mag) will be removed
    band : str
        Band to use for magnitude filtering
    verbose : bool
        Print filtering statistics

    Returns
    -------
    StarCatalog : Filtered catalog with bright stars removed
    """
    if star_catalog is None:
        return None

    n_original = len(star_catalog)

    if verbose:
        print(f"\n=== BRIGHT STAR FILTERING ===")
        print(f"Removing stars with mag < {mag_threshold} in {band}-band")
        print(f"Original stars: {n_original}")

    star_data = star_catalog._star_cat
    current_indices = star_catalog.indices

    mag_column = f'{band}_ab'
    if mag_column not in star_data.dtype.names:
        available_bands = [col.replace('_ab', '') for col in star_data.dtype.names if '_ab' in col]
        if available_bands:
            band = available_bands[0]
            mag_column = f'{band}_ab'
            if verbose:
                print(f"Using {band}-band instead")
        else:
            print("ERROR: No magnitude columns found!")
            return star_catalog

    magnitudes = star_data[mag_column][current_indices]

    keep_mask = np.isfinite(magnitudes) & (magnitudes >= mag_threshold)
    n_kept = np.sum(keep_mask)
    n_removed = n_original - n_kept

    if verbose:
        print(f"Stars kept (mag >= {mag_threshold}): {n_kept} ({100*n_kept/n_original:.1f}%)")
        print(f"Bright stars removed: {n_removed}")

    if n_kept == 0:
        print("WARNING: All stars would be removed! Using original catalog.")
        return star_catalog

    if n_removed == 0:
        return star_catalog

    filtered_catalog = StarCatalog(
        rng=star_catalog.rng,
        layout='random',
        coadd_dim=600,
        buff=0,
        pixel_scale=0.2,
        density=n_kept,
    )

    filtered_catalog._star_cat = star_catalog._star_cat
    filtered_catalog.shifts_array = star_catalog.shifts_array[keep_mask]
    filtered_catalog.indices = star_catalog.indices[keep_mask]
    filtered_catalog.density = star_catalog.density * (n_kept / n_original)

    return filtered_catalog


# =============================================================================
# Single Image Generation
# =============================================================================

def generate_single_image(
    rng, config, psf, layout, shifts, g1, g2, crop_size=None
):
    """
    Generate a single simulated image with catalog.

    Returns
    -------
    image : torch.Tensor
        Image tensor of shape [n_bands, height, width]
    positions : torch.Tensor
        Source positions tensor of shape [n_sources, 2]
    n_sources : int
        Number of sources in the image
    magnitude : array
        Magnitudes of sources
    """
    # Create galaxy catalog
    galaxy_catalog = WLDeblendGalaxyCatalog(
        rng=rng,
        coadd_dim=config['coadd_dim'],
        buff=config['buff'],
        layout=layout,
        sep=None,
        select_observable=config['select_observable'],
        select_lower_limit=config['select_lower_limit'],
        select_upper_limit=config['select_upper_limit']
    )

    galaxy_catalog.shifts_array = shifts
    num = len(galaxy_catalog.shifts_array)
    galaxy_catalog.indices = galaxy_catalog.rng.randint(
        0,
        galaxy_catalog._wldeblend_cat.size,
        size=num,
    )
    galaxy_catalog.angles = galaxy_catalog.rng.uniform(low=0, high=360, size=num)
    magnitude = galaxy_catalog._wldeblend_cat[galaxy_catalog.indices]["i_ab"]

    # Create star catalog if needed
    star_catalog = None
    if config['generate_stars']:
        star_catalog = make_star_catalog(
            rng=rng,
            coadd_dim=config['coadd_dim'],
            buff=config['buff'],
            pixel_scale=config['pixel_scale'],
            star_config=config.get('star_config')
        )

        star_setting = config.get('star_setting', {})
        if star_setting.get('star_filter', False):
            star_catalog = filter_bright_stars_production(
                star_catalog,
                mag_threshold=star_setting.get('star_filter_mag', 18),
                band=star_setting.get('star_filter_band', 'r'),
                verbose=False
            )

    # Run simulation
    sim_kwargs = dict(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=config['coadd_dim'],
        g1=g1,
        g2=g2,
        bands=config['bands'],
        psf=psf,
        noise_factor=config['noise_factor'],
        dither=config['dither'],
        dither_size=config['dither_size'],
        rotate=config['rotate'],
        cosmic_rays=config['cosmic_rays'],
        bad_columns=config['bad_columns'],
        star_bleeds=config['star_bleeds'],
    )

    if star_catalog is not None:
        sim_kwargs['star_catalog'] = star_catalog
        sim_kwargs['draw_bright'] = config['draw_bright']

    sim_data = make_sim(**sim_kwargs)

    # Extract truth info
    truth = sim_data['truth_info']
    x_pos = np.array(truth['image_x'], copy=True)
    y_pos = np.array(truth['image_y'], copy=True)

    positions = torch.stack([
        torch.from_numpy(x_pos),
        torch.from_numpy(y_pos)
    ], dim=1).float()

    n_sources = len(x_pos)

    # Extract images
    bands = config['bands']
    first_band = bands[0]
    h, w = sim_data['band_data'][first_band][0].image.array.shape

    image = torch.zeros(len(bands), h, w, dtype=torch.float32)
    for i, band in enumerate(bands):
        image_np = sim_data['band_data'][band][0].image.array
        image[i] = torch.from_numpy(image_np.copy())

    # Crop image if crop_size is specified
    if crop_size is not None:
        h_center, w_center = image.shape[1] // 2, image.shape[2] // 2
        half_crop = crop_size // 2
        image = image[:, h_center - half_crop:h_center + half_crop,
                         w_center - half_crop:w_center + half_crop]

    return image, positions, n_sources, magnitude


def process_single_image_worker(args):
    """
    Worker function for multiprocessing.
    Creates its own PSF to avoid serialization issues.
    Uses SeedSequence-spawned seeds for truly independent random streams.
    """
    (iter_idx, config, g1_val, g2_val, child_seed, layout, shifts) = args

    # Create independent RNG from spawned seed
    rng = np.random.RandomState(child_seed)

    # Create PSF in worker with its own seed (avoids serialization issues)
    se_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
    psf_seed = child_seed + 1000000  # Offset to avoid correlation with main rng
    psf = create_psf(config, se_dim, psf_seed)

    # Generate image (cropping is done inside generate_single_image)
    image, positions, n_sources, magnitude = generate_single_image(
        rng, config, psf, layout, shifts, g1_val, g2_val,
        crop_size=config.get('crop_size', 2048)
    )

    psf_param = get_optional_psf_param(psf, config)

    return iter_idx, image, positions, n_sources, magnitude, psf_param


# =============================================================================
# Batch Management Functions
# =============================================================================

def create_batch_tensors(batch_size, n_channels, img_size, max_sources_estimate):
    """
    Create empty tensors for a new batch.

    Returns a dict containing:
    - images: [batch_size, n_channels, img_size, img_size]
    - catalog: dict with locs, n_sources, shear_1, shear_2
    - psf_params: dict for optional PSF parameters
    """
    return {
        'images': torch.zeros(batch_size, n_channels, img_size, img_size, dtype=torch.float32),
        'catalog': {
            'locs': torch.full((batch_size, max_sources_estimate, 2), float('nan'), dtype=torch.float32),
            'n_sources': torch.zeros(batch_size, dtype=torch.long),
            'shear_1': torch.zeros(batch_size, dtype=torch.float32),
            'shear_2': torch.zeros(batch_size, dtype=torch.float32),
        },
        'psf_params': {}
    }


def add_image_to_batch(batch, local_idx, image, positions, n_sources, g1, g2, psf_param=None):
    """Add a single image and its data to the batch."""
    batch['images'][local_idx] = image

    # Handle case where we need to expand locs tensor
    max_sources = batch['catalog']['locs'].shape[1]
    if n_sources > max_sources:
        batch_size = batch['catalog']['locs'].shape[0]
        new_locs = torch.full((batch_size, n_sources, 2), float('nan'), dtype=torch.float32)
        new_locs[:, :max_sources] = batch['catalog']['locs']
        batch['catalog']['locs'] = new_locs

    batch['catalog']['locs'][local_idx, :n_sources] = positions
    batch['catalog']['n_sources'][local_idx] = n_sources
    batch['catalog']['shear_1'][local_idx] = float(g1)
    batch['catalog']['shear_2'][local_idx] = float(g2)

    if psf_param is not None:
        batch['psf_params'][local_idx] = psf_param


def save_batch(batch, batch_num, save_folder, save_psf=False):
    """
    Save batch tensors to disk.

    Returns True if all saves succeeded.
    """
    image_file = f"{save_folder}/batch_{batch_num}_images.pt"
    catalog_file = f"{save_folder}/batch_{batch_num}_catalog.pt"

    image_success = safe_save_tensor(batch['images'], image_file)
    catalog_success = safe_save_tensor(batch['catalog'], catalog_file)

    psf_success = True
    if save_psf and batch['psf_params']:
        psf_file = f"{save_folder}/batch_{batch_num}_psf_params.pt"
        psf_success = safe_save_tensor(batch['psf_params'], psf_file)

    return image_success and catalog_success and psf_success


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_images(config):
    """
    Main function to generate images in batches.

    Supports both sequential and multiprocessing modes.
    Saves each batch to disk immediately after generation.
    """
    num_images = config['num_images']
    batch_size = config['batch_size']
    save_folder = f"{config['output_dir']}/{config['setting']}"
    aggressive_gc = config.get('aggressive_gc', True)
    use_multiprocessing = config.get('use_multiprocessing', False)
    n_workers = config.get('n_workers', min(mp.cpu_count(), 8))
    save_psf = config.get('save_psf_param', False)

    # Set up environment
    os.environ['CATSIM_DIR'] = config['catsim_dir']
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n=== IMAGE GENERATION ===")
    print(f"Output folder: {save_folder}")
    print(f"Total images: {num_images}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {(num_images + batch_size - 1) // batch_size}")
    print(f"Mode: {'Multiprocessing' if use_multiprocessing else 'Sequential'}")
    if use_multiprocessing:
        print(f"Workers: {n_workers}")

    # Initialize RNG and pre-generate shear values
    rng = np.random.RandomState(config['seed'])

    # Draw shear components independently from N(0, 0.015^2)
    rng_shear = np.random.RandomState(config['seed'] + 1000)
    g1 = rng_shear.normal(0.0, 0.015, num_images).astype(np.float32)
    g2 = rng_shear.normal(0.0, 0.015, num_images).astype(np.float32)

    # Create independent seeds for each image using SeedSequence
    # This ensures truly independent random streams for multiprocessing
    ss = SeedSequence(config['seed'])
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(num_images)]

    # Create layout and get shifts
    layout = Layout(
        layout_name=config['layout_name'],
        coadd_dim=config['coadd_dim'],
        pixel_scale=config['pixel_scale'],
        buff=config['buff']
    )
    shifts = layout.get_shifts(rng, density=config['density'])

    # Create PSF for sequential mode
    se_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
    psf = create_psf(config, se_dim, config['seed']) if not use_multiprocessing else None

    # Estimate max sources for tensor allocation
    max_sources_estimate = int(config['density'] * 3)

    # Process batches
    successful_batches = 0
    total_batches = (num_images + batch_size - 1) // batch_size

    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_num = batch_start // batch_size + 1
        current_batch_size = batch_end - batch_start

        print(f"\n=== BATCH {batch_num}/{total_batches} ===")
        print(f"Images {batch_start} to {batch_end - 1}")

        # Create batch tensors
        batch = create_batch_tensors(
            current_batch_size,
            len(config['bands']),
            config.get('crop_size', 2048),
            max_sources_estimate
        )

        batch_indices = list(range(batch_start, batch_end))

        if use_multiprocessing:
            # Multiprocessing mode - use pre-spawned independent seeds
            args_list = []

            for global_idx in batch_indices:
                args_list.append((
                    global_idx, config, g1[global_idx], g2[global_idx],
                    child_seeds[global_idx], layout, shifts
                ))

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(process_single_image_worker, args): args[0]
                    for args in args_list
                }

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_idx),
                    total=len(batch_indices),
                    desc=f"Batch {batch_num}",
                    unit="img"
                ):
                    global_idx, image, positions, n_sources, mag, psf_param = future.result()
                    local_idx = global_idx - batch_start
                    add_image_to_batch(
                        batch, local_idx, image, positions, n_sources,
                        g1[global_idx], g2[global_idx], psf_param
                    )
        else:
            # Sequential mode - use child_seeds for reproducibility
            for local_idx, global_idx in enumerate(tqdm(batch_indices, desc=f"Batch {batch_num}")):
                # Create fresh RNG from child seed for each image
                img_rng = np.random.RandomState(child_seeds[global_idx])
                image, positions, n_sources, mag = generate_single_image(
                    img_rng, config, psf, layout, shifts, g1[global_idx], g2[global_idx],
                    crop_size=config.get('crop_size', 2048)
                )

                psf_param = get_optional_psf_param(psf, config)

                add_image_to_batch(
                    batch, local_idx, image, positions, n_sources,
                    g1[global_idx], g2[global_idx], psf_param
                )

                if aggressive_gc and local_idx % 5 == 0:
                    cleanup_memory()

        # Save batch
        if save_batch(batch, batch_num, save_folder, save_psf):
            successful_batches += 1
            print(f"Batch {batch_num} saved successfully")
        else:
            print(f"WARNING: Batch {batch_num} save failed")

        # Clean up
        del batch
        if aggressive_gc:
            cleanup_memory()

    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Successfully saved: {successful_batches}/{total_batches} batches")
    print(f"Output location: {save_folder}")

    return {
        'successful_batches': successful_batches,
        'total_batches': total_batches,
        'save_folder': save_folder
    }


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate weak lensing simulation images')
    parser.add_argument('--config', type=str, default='sim_config.yaml',
                        help='Path to config file (default: sim_config.yaml)')
    args = parser.parse_args()

    start_time = time.time()
    start_datetime = datetime.now()

    print(f"=== WEAK LENSING IMAGE GENERATION ===")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Load config
    config_file = args.config
    if not os.path.isabs(config_file):
        # Look for config relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, config_file)

    print(f"Loading config from: {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Save config snapshot
    config_dir = f"{config['output_dir']}/config_snapshots"
    os.makedirs(config_dir, exist_ok=True)

    timestamp = start_datetime.strftime('%Y%m%d_%H%M%S')
    snapshot_path = f"{config_dir}/config_{config['setting']}_{timestamp}.yaml"

    with open(snapshot_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config snapshot saved to: {snapshot_path}")

    # Run generation
    result = generate_images(config)

    end_time = time.time()
    print(f"\n=== COMPLETED ===")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()
