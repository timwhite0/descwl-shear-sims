import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from descwl_shear_sims.galaxies import FixedGalaxyCatalog
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.stars import StarCatalog, make_star_catalog, DEFAULT_STAR_CONFIG
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_shear_sims.psfs import make_ps_psf
from descwl_shear_sims.sim import get_se_dim
from tqdm import tqdm
from descwl_shear_sims.layout.layout import Layout
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time
from datetime import datetime
import concurrent.futures
import sys

os.environ['CATSIM_DIR'] = '/data/scratch/taodingr/lsst_stack/catsim' 

import shutil

save_folder = f"/data/scratch/taodingr/weak_lensing/descwl"
total, used, free = shutil.disk_usage(save_folder)
print(f"Disk space for {save_folder}:")
print(f"  Total: {total // (2**30)} GiB")
print(f"  Used:  {used // (2**30)} GiB")
print(f"  Free:  {free // (2**30)} GiB")
space_available = free // (2**30)

def get_existing_completion_status(save_folder, setting_prefix):
    """Check if the large tensor files exist and how many images they contain"""
    image_file = f"{save_folder}/images_{setting_prefix}.pt"
    catalog_file = f"{save_folder}/catalog_{setting_prefix}.pt"
    
    if os.path.exists(image_file) and os.path.exists(catalog_file):
        try:
            # Load just the shape info without loading entire tensor
            images = torch.load(image_file, weights_only=True, map_location='cpu')
            num_existing = images.shape[0]
            del images  # Free memory immediately
            return num_existing
        except Exception as e:
            print(f"Error reading existing files: {e}")
            return 0
    return 0

class LargeTensorManager:
    """
    Manages large tensors for accumulated images and catalogs with incremental saving
    """
    def __init__(self, config, save_folder):
        self.config = config
        self.save_folder = save_folder
        self.setting = config['setting']
        
        # File paths
        self.image_file = f"{save_folder}/images_{self.setting}.pt"
        self.catalog_file = f"{save_folder}/catalog_{self.setting}.pt"
        
        # Tensors will be initialized when we know the dimensions
        self.images_tensor = None
        self.catalog_tensor = None
        
        # Track progress
        self.current_idx = 0
        self.total_images = config['num_data']
        
        # Create save folder
        os.makedirs(save_folder, exist_ok=True)
        
    def initialize_tensors(self, sample_image, sample_positions, sample_M, sample_g1, sample_g2):
        """Initialize large tensors based on first sample"""
        print("Initializing large tensors...")
        
        # Image tensor dimensions: [num_images, channels, height, width]
        num_images = self.total_images
        channels, height, width = sample_image.shape
        
        print(f"Image tensor shape: [{num_images}, {channels}, {height}, {width}]")
        
        # Initialize image tensor
        self.images_tensor = torch.zeros(num_images, channels, height, width, dtype=torch.float32)
        
        # For catalog, we need to determine max sources across all images
        # We'll use a conservative estimate and pad as needed
        max_sources_estimate = sample_M * 3  # Conservative estimate
        
        print(f"Estimated max sources per image: {max_sources_estimate}")
        
        # Catalog tensor structure - similar to your original but for all images at once
        self.catalog_tensor = {
            "locs": torch.full((num_images, max_sources_estimate, 2), float('nan'), dtype=torch.float32),
            "n_sources": torch.zeros(num_images, dtype=torch.long),
            "shear_1": torch.zeros(num_images, max_sources_estimate, 1, dtype=torch.float32),
            "shear_2": torch.zeros(num_images, max_sources_estimate, 1, dtype=torch.float32),
        }
        
        print("Large tensors initialized successfully!")
        
    def add_image_data(self, idx, image, positions, n_sources, g1, g2):
        """Add a single image and its catalog data to the large tensors"""
        if self.images_tensor is None:
            self.initialize_tensors(image, positions, n_sources, g1, g2)
        
        # Add image (remove batch dimension if present)
        if len(image.shape) == 4:
            image = image.squeeze(0)
        self.images_tensor[idx] = image
        
        # Add catalog data
        actual_sources = positions.shape[0]
        max_sources = self.catalog_tensor["locs"].shape[1]
        
        # Expand tensors if needed
        if actual_sources > max_sources:
            print(f"Expanding catalog tensors from {max_sources} to {actual_sources} sources")
            self._expand_catalog_tensors(actual_sources)
            max_sources = actual_sources
        
        # Fill in the data
        self.catalog_tensor["locs"][idx, :actual_sources] = positions
        self.catalog_tensor["n_sources"][idx] = n_sources
        
        # Broadcast shear values to all sources for this image
        if actual_sources > 0:
            # Convert numpy scalars to torch tensors if needed
            g1_tensor = torch.tensor(g1, dtype=torch.float32) if not isinstance(g1, torch.Tensor) else g1
            g2_tensor = torch.tensor(g2, dtype=torch.float32) if not isinstance(g2, torch.Tensor) else g2
            
            self.catalog_tensor["shear_1"][idx, :actual_sources, 0] = g1_tensor
            self.catalog_tensor["shear_2"][idx, :actual_sources, 0] = g2_tensor
        
        self.current_idx = max(self.current_idx, idx + 1)
        
    def _expand_catalog_tensors(self, new_max_sources):
        """Expand catalog tensors to accommodate more sources"""
        old_max = self.catalog_tensor["locs"].shape[1]
        num_images = self.catalog_tensor["locs"].shape[0]
        
        # Create new tensors with expanded size
        new_locs = torch.full((num_images, new_max_sources, 2), float('nan'), dtype=torch.float32)
        new_shear_1 = torch.zeros(num_images, new_max_sources, 1, dtype=torch.float32)
        new_shear_2 = torch.zeros(num_images, new_max_sources, 1, dtype=torch.float32)
        
        # Copy existing data
        new_locs[:, :old_max] = self.catalog_tensor["locs"]
        new_shear_1[:, :old_max] = self.catalog_tensor["shear_1"]
        new_shear_2[:, :old_max] = self.catalog_tensor["shear_2"]
        
        # Update catalog tensor
        self.catalog_tensor["locs"] = new_locs
        self.catalog_tensor["shear_1"] = new_shear_1
        self.catalog_tensor["shear_2"] = new_shear_2
        
    def save_after_each_image(self, verbose=False):
        """Save current state of tensors after each image - overwrites previous save"""
        if self.images_tensor is None:
            return
            
        if verbose:
            print(f"💾 Saving progress: {self.current_idx} images", flush=True)
        
        # Save only the filled portion of tensors (up to current_idx)
        current_images = self.images_tensor[:self.current_idx]
        current_catalog = {
            "locs": self.catalog_tensor["locs"][:self.current_idx],
            "n_sources": self.catalog_tensor["n_sources"][:self.current_idx],
            "shear_1": self.catalog_tensor["shear_1"][:self.current_idx],
            "shear_2": self.catalog_tensor["shear_2"][:self.current_idx],
        }
        
        # Save with compression and immediate disk sync
        torch.save(current_images, self.image_file, _use_new_zipfile_serialization=True)
        torch.save(current_catalog, self.catalog_file, _use_new_zipfile_serialization=True)
        
        # Force immediate write to disk
        try:
            with open(self.image_file, 'r+b') as f:
                f.flush()
                os.fsync(f.fileno())
            
            with open(self.catalog_file, 'r+b') as f:
                f.flush()
                os.fsync(f.fileno())
        except:
            pass
            
        if verbose:
            print(f"✅ Saved {self.current_idx} images to disk", flush=True)
    
    def save_incremental(self, force_save=False):
        """Save current state of tensors (with progress messages every N images)"""
        if self.images_tensor is None:
            return
            
        # Always save after each image, but only show progress every 50 images
        self.save_after_each_image(verbose=(self.current_idx % 50 == 0 or force_save))
                
    def finalize_save(self):
        """Final save of all accumulated data"""
        print(f"Finalizing save: {self.current_idx} total images")
        
        if self.images_tensor is None:
            print("No data to save!")
            return
            
        # Save final tensors (only filled portion)
        final_images = self.images_tensor[:self.current_idx]
        final_catalog = {
            "locs": self.catalog_tensor["locs"][:self.current_idx],
            "n_sources": self.catalog_tensor["n_sources"][:self.current_idx],
            "shear_1": self.catalog_tensor["shear_1"][:self.current_idx],
            "shear_2": self.catalog_tensor["shear_2"][:self.current_idx],
        }
        
        torch.save(final_images, self.image_file, _use_new_zipfile_serialization=True)
        torch.save(final_catalog, self.catalog_file, _use_new_zipfile_serialization=True)
        
        print(f"✅ Final save completed!")
        print(f"📁 Images saved: {self.image_file}")
        print(f"📁 Catalog saved: {self.catalog_file}")
        print(f"📊 Final tensor shapes:")
        print(f"   Images: {final_images.shape}")
        print(f"   Locations: {final_catalog['locs'].shape}")
        print(f"   Sources per image: {final_catalog['n_sources'].shape}")

def filter_bright_stars_production(star_catalog, mag_threshold=18, band='r', verbose=True):
    """
    Production-ready bright star filter for massive performance gains
    
    Based on test results:
    - mag_threshold=18: 91.6% faster (removes stars that go through draw_bright_star)
    - mag_threshold=15: 29.4% faster (removes only very bright stars)
    
    Parameters:
    -----------
    star_catalog : StarCatalog
        Original star catalog from make_star_catalog()
    mag_threshold : float
        Magnitude threshold - stars brighter (lower mag) than this will be removed
        Recommended values:
        - 18.0: Maximum speedup (91.6% faster)
        - 15.0: Conservative speedup (29.4% faster) 
        - 16.0: Moderate speedup (49.9% faster)
    band : str
        Band to use for magnitude filtering (default: 'r')
    verbose : bool
        Print filtering statistics
    """
    if star_catalog is None:
        return None
    
    n_original = len(star_catalog)
    
    if verbose:
        print(f"\n=== BRIGHT STAR FILTERING (PRODUCTION) ===")
        print(f"Removing stars with mag < {mag_threshold} in {band}-band")
        print(f"Original stars: {n_original}")
    
    # Get star data and current indices
    star_data = star_catalog._star_cat
    current_indices = star_catalog.indices
    
    # Get magnitudes for the current stars
    mag_column = f'{band}_ab'
    if mag_column not in star_data.dtype.names:
        available_bands = [col.replace('_ab', '') for col in star_data.dtype.names if '_ab' in col]
        if available_bands:
            band = available_bands[0]
            mag_column = f'{band}_ab'
            if verbose:
                print(f"Using {band}-band instead of {band}")
        else:
            print(f"ERROR: No magnitude columns found!")
            return star_catalog
    
    magnitudes = star_data[mag_column][current_indices]
    
    # Create mask to keep only non-bright stars (mag >= threshold)
    # Lower magnitude = brighter star, so we keep stars with mag >= threshold
    keep_mask = np.isfinite(magnitudes) & (magnitudes >= mag_threshold)
    n_kept = np.sum(keep_mask)
    n_removed = n_original - n_kept
    
    if verbose:
        print(f"Stars kept (mag >= {mag_threshold}): {n_kept} ({100*n_kept/n_original:.1f}%)")
        print(f"Bright stars removed (mag < {mag_threshold}): {n_removed} ({100*n_removed/n_original:.1f}%)")
    
    if n_kept == 0:
        print("WARNING: All stars would be removed! Using original catalog.")
        return star_catalog
    
    if n_removed == 0:
        print("INFO: No bright stars found to remove.")
        return star_catalog
    
    # Create filtered catalog
    filtered_catalog = StarCatalog(
        rng=star_catalog.rng,
        layout='random',  # Will be overridden
        coadd_dim=600,    # Will be overridden  
        buff=0,           # Will be overridden
        pixel_scale=0.2,  # Will be overridden
        density=n_kept,   # Approximate density
    )
    
    # Copy filtered data
    filtered_catalog._star_cat = star_catalog._star_cat
    filtered_catalog.shifts_array = star_catalog.shifts_array[keep_mask]
    filtered_catalog.indices = star_catalog.indices[keep_mask]
    filtered_catalog.density = star_catalog.density * (n_kept / n_original)
    
    if verbose:
        print(f"✅ Filtered catalog created: {len(filtered_catalog)} stars")
        print(f"🚀 Expected to prevent {n_removed} expensive draw_bright_star() calls")
    
    return filtered_catalog

def estimate_tensor_size_mb(tensor):
    """
    Estimate the size of a tensor in MB when saved to disk
    """
    # Calculate the size in bytes
    element_size = tensor.element_size()  # Size of each element in bytes
    num_elements = tensor.numel()  # Total number of elements
    size_bytes = element_size * num_elements
    
    # Add overhead for PyTorch's .pt format (approximately 10-20% overhead)
    overhead_factor = 1.15
    estimated_size_bytes = size_bytes * overhead_factor
    
    # Convert to MB
    size_mb = estimated_size_bytes / (1024 * 1024)
    return size_mb

def check_storage_requirements_large_tensors(config, sample_image, sample_catalog_dict, save_folder):
    """
    Estimate storage requirements for large tensor approach
    """
    num_total_images = config['num_data']
    
    # Estimate final tensor sizes
    channels, height, width = sample_image.shape
    max_sources = sample_catalog_dict['locs'].shape[0] * 3  # Conservative estimate
    
    # Create dummy tensors to estimate size
    dummy_image_tensor = torch.zeros(num_total_images, channels, height, width, dtype=torch.float32)
    dummy_catalog = {
        "locs": torch.zeros(num_total_images, max_sources, 2, dtype=torch.float32),
        "n_sources": torch.zeros(num_total_images, dtype=torch.long),
        "shear_1": torch.zeros(num_total_images, max_sources, 1, dtype=torch.float32),
        "shear_2": torch.zeros(num_total_images, max_sources, 1, dtype=torch.float32),
    }
    
    # Estimate sizes
    image_size_mb = estimate_tensor_size_mb(dummy_image_tensor)
    
    catalog_size_mb = 0
    for key, tensor in dummy_catalog.items():
        catalog_size_mb += estimate_tensor_size_mb(tensor)
    
    total_estimated_mb = image_size_mb + catalog_size_mb
    total_estimated_gb = total_estimated_mb / 1024
    
    print(f"\n--- Large Tensor Storage Estimation ---")
    print(f"Final image tensor size: {image_size_mb:.2f} MB")
    print(f"Final catalog tensors size: {catalog_size_mb:.2f} MB")
    print(f"Total estimated size: {total_estimated_gb:.2f} GB")
    
    # Check available disk space
    total, used, free = shutil.disk_usage(save_folder)
    free_gb = free / (1024**3)
    
    print(f"Available disk space: {free_gb:.2f} GB")
    
    # Add 20% safety margin for large tensors (they need more temp space)
    safety_margin = 1.2
    required_space_gb = total_estimated_gb * safety_margin
    
    if required_space_gb > free_gb:
        print(f"❌ WARNING: Not enough disk space!")
        print(f"Required space (with 20% margin): {required_space_gb:.2f} GB")
        print(f"Available space: {free_gb:.2f} GB")
        print(f"Shortfall: {required_space_gb - free_gb:.2f} GB")
        return False
    else:
        print(f"✅ Sufficient disk space available")
        print(f"Required space (with 20% margin): {required_space_gb:.2f} GB")
        return True

def Generate_single_img_catalog(
    ntrial, rng, mag, hlr, psf, morph, pixel_scale, layout, coadd_dim, buff, sep, g1, g2, bands, 
    noise_factor, dither, dither_size, rotate, cosmic_rays, bad_columns, star_bleeds, star_catalog, shifts,
    catalog_type, select_observable, select_lower_limit, select_upper_limit
    ):
    """
    Generate one catalog and image - optimized version
    """
    # Remove unnecessary loop - ntrial should always be 1 for single image generation
    if ntrial != 1:
        print("Warning: ntrial should be 1 for single image generation")

    if catalog_type == 'wldeblend':
        galaxy_catalog = WLDeblendGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout=layout,
        sep=sep,
        select_observable=select_observable,
        select_lower_limit=select_lower_limit,
        select_upper_limit=select_upper_limit
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
    else:
        galaxy_catalog = FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout=layout,
        mag=mag,
        hlr=hlr,
        morph=morph,
        pixel_scale=pixel_scale,
        sep=sep
        )

        galaxy_catalog.shifts_array = shifts
        magnitude = []

    if star_catalog is None:
        # only generate galaxies
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            g1=g1,
            g2=g2,
            bands=bands,
            psf=psf,
            noise_factor=noise_factor,
            dither=dither,
            dither_size=dither_size,
            rotate=rotate,
            cosmic_rays=cosmic_rays,
            bad_columns=bad_columns,
            star_bleeds=star_bleeds
        )
    else:
        # generate galaxies and stars
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            star_catalog=star_catalog,
            coadd_dim=coadd_dim,
            g1=g1,
            g2=g2,
            bands=bands,
            psf=psf,
            noise_factor=noise_factor,
            dither=dither,
            dither_size=dither_size,
            rotate=rotate,
            cosmic_rays=cosmic_rays,
            bad_columns=bad_columns,
            star_bleeds=star_bleeds,
            draw_bright=False,
        )

    # get the truth (source) information
    truth = sim_data['truth_info']
    image_x_positions = truth['image_x']
    image_y_positions = truth['image_y']
    
    # Fix numpy array memory layout issues for multiprocessing
    # Copy arrays to ensure contiguous memory layout
    x_pos_copy = np.array(image_x_positions, copy=True)
    y_pos_copy = np.array(image_y_positions, copy=True)
    
    positions_tensor = torch.stack([
        torch.from_numpy(x_pos_copy),
        torch.from_numpy(y_pos_copy)
    ], dim=1).float()
    
    M = len(image_x_positions)

    # Pre-allocate list for better performance
    n_bands = len(bands)
    first_band = bands[0]
    h, w = sim_data['band_data'][first_band][0].image.array.shape
    
    # Pre-allocate tensor
    image_tensor = torch.zeros(n_bands, h, w, dtype=torch.float32)
    
    for i, band in enumerate(bands):
        image_np = sim_data['band_data'][band][0].image.array
        image_tensor[i] = torch.from_numpy(image_np.copy())
    
    return image_tensor, positions_tensor, M, magnitude

def process_single_image_for_accumulation(args):
    """Function to process a single image for accumulation into large tensors"""
    (iter_idx, config, g1_val, g2_val, rng_state, psf, layout, shifts, star_catalog) = args
    
    # Create new RNG with the passed state
    rng = np.random.RandomState()
    rng.set_state(rng_state)
    # Advance the RNG state for this iteration to ensure uniqueness
    for _ in range(iter_idx):
        rng.rand()
    
    each_image, positions_tensor, M, magnitude = Generate_single_img_catalog(
        1, rng, config['mag'], config['hlr'], psf, config['morph'], 
        config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
        config['sep'], g1_val, g2_val, config['bands'], config['noise_factor'], 
        config['dither'], config['dither_size'], config['rotate'], 
        config['cosmic_rays'], config['bad_columns'], config['star_bleeds'], 
        star_catalog, shifts, config['catalog_type'], config['select_observable'], 
        config['select_lower_limit'], config['select_upper_limit']
    )
    
    crop_size = 2048
    h_center, w_center = each_image.shape[1] // 2, each_image.shape[2] // 2
    half_crop = crop_size // 2
    
    single_image = each_image[:, 
                             h_center - half_crop:h_center + half_crop,
                             w_center - half_crop:w_center + half_crop]
    
    return iter_idx, single_image, positions_tensor, M, magnitude

# Replace your Generate_img_catalog function with this modified version
def Generate_img_catalog(config, use_multiprocessing=False, use_threading=False, n_workers=None, resume_mode=False, start_from=0, stop_on_multiprocessing_fail=True):
    """
    Generate a number of catalogs and images using large tensor accumulation
    
    Args:
        use_multiprocessing: Use ProcessPoolExecutor (fastest but may have serialization issues)
        use_threading: Use ThreadPoolExecutor (good compromise, but may have thread safety issues)
        If both are False, uses optimized sequential processing (recommended for stability)
        resume_mode: If True, check for existing files and only generate missing ones
        start_from: Start generating from this index (only used if resume_mode=False)
    """
    num_data = config['num_data']
    save_folder = f"/data/scratch/taodingr/weak_lensing/descwl/{config['setting']}"
    
    # Initialize tensor manager
    tensor_manager = LargeTensorManager(config, save_folder)
    
    # Resume mode: check existing completion
    if resume_mode:
        existing_count = get_existing_completion_status(save_folder, config['setting'])
        print(f"Found {existing_count} existing images in large tensors")
        
        if existing_count >= num_data:
            print("✅ All images already exist! Nothing to generate.")
            return [[] for _ in range(num_data)]
        
        print(f"Need to generate {num_data - existing_count} more images")
        indices_to_generate = list(range(existing_count, num_data))
        
        # Load existing tensors if they exist
        if existing_count > 0:
            print("Loading existing tensors...")
            try:
                existing_images = torch.load(tensor_manager.image_file, weights_only=True, map_location='cpu')
                existing_catalog = torch.load(tensor_manager.catalog_file, weights_only=True, map_location='cpu')
                
                # Initialize manager with existing data
                tensor_manager.images_tensor = torch.zeros(num_data, *existing_images.shape[1:], dtype=torch.float32)
                tensor_manager.images_tensor[:existing_count] = existing_images
                
                tensor_manager.catalog_tensor = {
                    "locs": torch.full((num_data, existing_catalog["locs"].shape[1], 2), float('nan'), dtype=torch.float32),
                    "n_sources": torch.zeros(num_data, dtype=torch.long),
                    "shear_1": torch.zeros(num_data, existing_catalog["shear_1"].shape[1], 1, dtype=torch.float32),
                    "shear_2": torch.zeros(num_data, existing_catalog["shear_2"].shape[1], 1, dtype=torch.float32),
                }
                
                # Copy existing data
                tensor_manager.catalog_tensor["locs"][:existing_count] = existing_catalog["locs"]
                tensor_manager.catalog_tensor["n_sources"][:existing_count] = existing_catalog["n_sources"]
                tensor_manager.catalog_tensor["shear_1"][:existing_count] = existing_catalog["shear_1"]
                tensor_manager.catalog_tensor["shear_2"][:existing_count] = existing_catalog["shear_2"]
                
                tensor_manager.current_idx = existing_count
                print(f"✅ Loaded existing data: {existing_count} images")
                
                # Clean up temporary tensors
                del existing_images, existing_catalog
                
            except Exception as e:
                print(f"Error loading existing tensors: {e}")
                print("Starting fresh...")
                indices_to_generate = list(range(num_data))
    else:
        # Regular mode: generate from start_from to end
        indices_to_generate = list(range(start_from, num_data))
        if start_from > 0:
            print(f"Starting generation from index {start_from}")
    
    if not indices_to_generate:
        print("No images to generate.")
        return []
    
    rng = np.random.RandomState(config['seed'])
    input_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
    print(f"The input image dimension is {input_dim}")
    
    # Pre-generate all shear values at once (for ALL possible indices)
    if config['shear_setting'] == "const":
        const = 0.05
        g1 = np.full(num_data, const, dtype=np.float32)
        g2 = np.full(num_data, const, dtype=np.float32)
    elif config['shear_setting'] == "vary":
        # Draw from Gaussian and clip in one go
        rng_shear = np.random.RandomState(config['seed'] + 1000)  # Separate RNG for shear to ensure consistency
        g1 = np.clip(rng_shear.normal(0.0, 0.015, num_data), -0.05, 0.05).astype(np.float32)
        g2 = np.clip(rng_shear.normal(0.0, 0.015, num_data), -0.05, 0.05).astype(np.float32)
    
    # Create star catalog once if needed
    star_catalog = None
    if config['star_catalog'] is not None:
        star_catalog = make_star_catalog(
            rng=rng,
            coadd_dim=config['coadd_dim'],
            buff=config['buff'],
            pixel_scale=config['pixel_scale'],
            star_config=config['star_config']
        )
        print("Generating Galaxies and Stars")
        if config['star_filter']:
            star_catalog = filter_bright_stars_production(
                star_catalog, 
                mag_threshold=config['star_filter_mag'], 
                band=config['star_filter_band'], 
                verbose=True)
            if star_catalog is not None:
                print(f"✅ Generating Galaxies with filtered Star Catalog (mag >= {config['star_filter_mag']})")
            else:
                print("⚠️  All bright stars filtered. Generating only Galaxies.")
        else:
            print("✅ Generating Galaxies with original Star Catalog")
    else:
        print("Only generating Galaxies")
    
    # Create PSF once
    if config['psf_type'] == "gauss":
        psf = make_fixed_psf(psf_type=config['psf_type']) 
    elif config['psf_type'] == "moffat":  
        psf = make_fixed_psf(psf_type=config['psf_type'], psf_fwhm=0.8)
    
    # Create layout once
    layout = Layout(
        layout_name=config['layout_name'],
        coadd_dim=config['coadd_dim'],
        pixel_scale=config['pixel_scale'],
        buff=config['buff']
    )
    
    shifts = layout.get_shifts(rng, density=config['density'])

    # Storage check only if starting fresh
    if tensor_manager.current_idx == 0:
        # Generate first image to estimate storage requirements
        print("Generating first image to estimate storage requirements...")
        first_idx = indices_to_generate[0]
        first_image, first_positions, first_M, first_magnitude = Generate_single_img_catalog(
            1, rng, config['mag'], config['hlr'], psf, config['morph'], 
            config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
            config['sep'], g1[first_idx], g2[first_idx], config['bands'], config['noise_factor'], 
            config['dither'], config['dither_size'], config['rotate'], 
            config['cosmic_rays'], config['bad_columns'], config['star_bleeds'], 
            star_catalog, shifts, config['catalog_type'], config['select_observable'], 
            config['select_lower_limit'], config['select_upper_limit']
        )
       
        # Crop first image
        H, W = first_image.shape[1], first_image.shape[2]
        crop_size = 2048
        start_h = (H - crop_size) // 2
        start_w = (W - crop_size) // 2
        first_image_cropped = first_image[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
        
        # Create first catalog dict for estimation
        first_catalog_dict = {
            "locs": first_positions,
            "n_sources": first_M,
            "shear_1": g1[first_idx],
            "shear_2": g2[first_idx],
        }
        
        # Check storage requirements for large tensors
        if not check_storage_requirements_large_tensors(config, first_image_cropped, first_catalog_dict, save_folder):
            print("❌ STOPPING: Insufficient disk space for large tensors!")
            print("Consider:")
            print("1. Reducing num_data in config")
            print("2. Using a different storage location")
            print("3. Freeing up disk space")
            return None
        
        # Initialize tensor manager with first image
        tensor_manager.add_image_data(first_idx, first_image_cropped, first_positions, first_M, g1[first_idx], g2[first_idx])
        # Save immediately after first image
        tensor_manager.save_after_each_image(verbose=True)
        print(f"✅ Added first image (index {first_idx}) to large tensors")
        
        # Remove first index from generation list since we already processed it
        indices_to_generate = indices_to_generate[1:]
    
    # If we reach here, we have enough space - continue with generation
    print(f"✅ Sufficient storage available. Generating {len(indices_to_generate)} images...")
    
    magnitudes = []

    # Generate the remaining images
    if len(indices_to_generate) > 0:
        # Try multiprocessing first (if enabled)
        if use_multiprocessing and len(indices_to_generate) > 0:
            if n_workers is None:
                n_workers = min(mp.cpu_count(), len(indices_to_generate), 8)
            
            print(f"Attempting multiprocessing with {n_workers} workers for {len(indices_to_generate)} images")
            
            try:
                args_list = []
                rng_state = rng.get_state()
                for iter_idx in indices_to_generate:
                    args_list.append((
                        iter_idx, config, g1[iter_idx], g2[iter_idx], 
                        rng_state, psf, layout, shifts, star_catalog
                    ))
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all tasks and get futures
                    future_to_idx = {executor.submit(process_single_image_for_accumulation, args): args[0] for args in args_list}
    
                    completed_count = 0
    
                    # Create tqdm progress bar with more info
                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_idx),
                        total=len(indices_to_generate),
                        desc="Generating images (multiprocessing)",
                        unit="image",
                        ncols=120,
                        leave=True
                    ):
                        iter_idx, single_image, positions_tensor, M, magnitude = future.result()
            
                        # Add to large tensors immediately
                        tensor_manager.add_image_data(iter_idx, single_image, positions_tensor, M, g1[iter_idx], g2[iter_idx])
                        
                        # Save after EACH image (overwrites previous save)
                        tensor_manager.save_after_each_image(verbose=False)
            
                        if single_image.shape[1] != 2048:
                            print("The output dimension is not 2048, check coadd_dim and rotate")
                            break
            
                        magnitudes.append(magnitude)
                        completed_count += 1

            except Exception as e:
                print(f"\n❌ Multiprocessing failed: {e}")
    
                if stop_on_multiprocessing_fail:
                    print("🛑 STOPPING execution as requested (stop_on_multiprocessing_fail=True)")
                    print(f"✅ Successfully completed: {len(magnitudes)} images")
                    print("💡 To resume from where it left off, run again with resume_mode=True")
                    print("💡 To use fallback methods instead, set stop_on_multiprocessing_fail=False in config")
        
                    # Save current progress before stopping
                    tensor_manager.finalize_save()
                    return magnitudes  # Return what we have so far
                else:
                    print("⚠️  Falling back to threading/sequential as configured...")
                    use_threading = True
                    use_multiprocessing = False
        
        if use_threading and len(indices_to_generate) > len(magnitudes) and not use_multiprocessing:
            # Threading approach - good compromise
            if n_workers is None:
                n_workers = min(4, len(indices_to_generate))  # Conservative for threading
            
            print(f"Using threading with {n_workers} workers for {len(indices_to_generate)} images")
            
            def process_with_threading(iter_idx):
                print(f"🔄 Generating image {iter_idx}...", flush=True)
                
                thread_rng = np.random.RandomState(config['seed'] + iter_idx * 1000)
                thread_layout = layout
                thread_psf = psf
                thread_shifts = shifts if shifts is not None else layout.get_shifts(thread_rng, density=config['density'])
                thread_star_catalog = star_catalog
          
                each_image, positions_tensor, M, magnitude = Generate_single_img_catalog(
                    1, thread_rng, config['mag'], config['hlr'], thread_psf, config['morph'], 
                    config['pixel_scale'], thread_layout, config['coadd_dim'], config['buff'], 
                    config['sep'], g1[iter_idx], g2[iter_idx], config['bands'], 
                    config['noise_factor'], config['dither'], config['dither_size'], 
                    config['rotate'], config['cosmic_rays'], config['bad_columns'], 
                    config['star_bleeds'], thread_star_catalog, thread_shifts,
                    config['catalog_type'], config['select_observable'], config['select_lower_limit'], 
                    config['select_upper_limit']
                )
 
                # Crop image
                H, W = each_image.shape[1], each_image.shape[2]
                crop_size = 2048
                start_h = (H - crop_size) // 2
                start_w = (W - crop_size) // 2
                single_image = each_image[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
        
                print(f"✅ Generated image {iter_idx}", flush=True)
                return iter_idx, single_image, positions_tensor, M, magnitude
            
            # Process tasks and accumulate into tensors as they complete
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                future_to_idx = {executor.submit(process_with_threading, idx): idx for idx in indices_to_generate}
            
                completed_count = 0
                # Process each task as it completes
                for future in concurrent.futures.as_completed(future_to_idx):
                    iter_idx, single_image, positions_tensor, M, magnitude = future.result()
                
                    # Add to large tensors immediately
                    tensor_manager.add_image_data(iter_idx, single_image, positions_tensor, M, g1[iter_idx], g2[iter_idx])
                    
                    # Save after EACH image (overwrites previous save)
                    tensor_manager.save_after_each_image(verbose=(completed_count % 50 == 0))
                
                    magnitudes.append(magnitude)
                    completed_count += 1
                
                    print(f"📊 Progress: {completed_count}/{len(indices_to_generate)} images accumulated", flush=True)
                
                    if single_image.shape[1] != 2048:
                        print("The output dimension is not 2048, check coadd_dim and rotate")
                        break
        
        elif not use_multiprocessing and not use_threading and len(indices_to_generate) > len(magnitudes):
            # Sequential processing
            for iter_idx in tqdm(indices_to_generate, desc="Generating images (sequential)"):
                each_image, positions_tensor, M, magnitude = Generate_single_img_catalog(
                    1, rng, config['mag'], config['hlr'], psf, config['morph'], 
                    config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
                    config['sep'], g1[iter_idx], g2[iter_idx], config['bands'], 
                    config['noise_factor'], config['dither'], config['dither_size'], 
                    config['rotate'], config['cosmic_rays'], config['bad_columns'], 
                    config['star_bleeds'], star_catalog, shifts, 
                    config['catalog_type'], config['select_observable'], 
                    config['select_lower_limit'], config['select_upper_limit']
                )
                
                # Optimized cropping
                H, W = each_image.shape[1], each_image.shape[2]
                crop_size = 2048
                
                start_h = (H - crop_size) // 2
                start_w = (W - crop_size) // 2
                
                single_image = each_image[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
                
                if single_image.shape[1] != 2048:
                    print("The output dimension is not 2048, check coadd_dim and rotate")
                    break
                
                # Add to large tensors
                tensor_manager.add_image_data(iter_idx, single_image, positions_tensor, M, g1[iter_idx], g2[iter_idx])
                
                # Save after EACH image (overwrites previous save)
                tensor_manager.save_after_each_image(verbose=(iter_idx % 50 == 0))
                
                magnitudes.append(magnitude)

    # Final save of all accumulated data
    tensor_manager.finalize_save()
    
    return magnitudes

def plot_magnitude_distribution(magnitudes, num_selected, config):
    if num_selected > config['num_data']:
        print(f"selected number of galaxies is greater than generated number of galaxies, setting num_selected to {config['num_data']}")
        num_selected = config['num_data']

    mag = magnitudes[:num_selected]
    mag_combined = np.concatenate(mag)
    plt.hist(mag_combined, bins=100)
    plt.xlabel("i-band ab magnitude")
    plt.ylabel("Count")
    plt.savefig(f"/data/scratch/taodingr/lsst_stack/descwl-shear-sims/notebooks/magnitude_distribution.png")

def load_large_tensor_data(setting, data_folder="/data/scratch/taodingr/weak_lensing/descwl"):
    """
    Convenience function to load the large tensor data
    
    Returns:
        images: torch.Tensor of shape [num_images, channels, height, width]
        catalog: dict with keys 'locs', 'n_sources', 'shear_1', 'shear_2'
    """
    save_folder = f"{data_folder}/{setting}"
    image_file = f"{save_folder}/images_{setting}.pt"
    catalog_file = f"{save_folder}/catalog_{setting}.pt"
    
    print(f"Loading images from: {image_file}")
    print(f"Loading catalog from: {catalog_file}")
    
    # Load tensors
    images = torch.load(image_file, weights_only=True, map_location='cpu')
    catalog = torch.load(catalog_file, weights_only=True, map_location='cpu')
    
    print(f"✅ Loaded successfully!")
    print(f"Images shape: {images.shape}")
    print(f"Number of images: {len(catalog['n_sources'])}")
    print(f"Catalog keys: {list(catalog.keys())}")
    
    return images, catalog

def main():
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"=== LARGE TENSOR SIMULATION STARTED ===")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 50)
    with open('sim_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Add resume capability options
    resume_mode = config.get('resume_mode', True)  # Default to True for resumable generation
    start_from = config.get('start_from', 0)  # Only used if resume_mode=False
    
    # Add option to control multiprocessing and threading
    use_multiprocessing = config.get('use_multiprocessing', False)  # Default to False due to serialization issues
    use_threading = config.get('use_threading', True)  # Default to True as it's more stable
    n_workers = config.get('n_workers', None)
    
    processing_mode = "Sequential"
    if use_multiprocessing:
        processing_mode = "Multiprocessing"
    elif use_threading:
        processing_mode = "Threading"
    
    print(f"Processing mode: {processing_mode}")
    print(f"Resume mode: {'Enabled' if resume_mode else 'Disabled'}")
    print(f"Large tensor accumulation: ENABLED")
    if not resume_mode and start_from > 0:
        print(f"Starting from index: {start_from}")
    
    stop_on_multiprocessing_fail = config.get('stop_on_multiprocessing_fail', True)

    magnitudes = Generate_img_catalog(
        config, use_multiprocessing=use_multiprocessing, 
        use_threading=use_threading, n_workers=n_workers,
        resume_mode=resume_mode, start_from=start_from,
        stop_on_multiprocessing_fail=stop_on_multiprocessing_fail 
    )
    
    # Check if generation was stopped due to insufficient storage
    if magnitudes is None:
        print("Image generation stopped due to insufficient disk space.")
        return

    # Load and verify the final results
    save_folder = f"/data/scratch/taodingr/weak_lensing/descwl/{config['setting']}"
    if os.path.exists(save_folder):
        try:
            # Try loading the large tensors to verify
            images, catalog = load_large_tensor_data(config['setting'])
            
            print(f"\n=== FINAL VERIFICATION ===")
            print(f"Total images generated: {images.shape[0]}")
            print(f"Image dimensions: {images.shape[1:]} (channels, height, width)")
            print(f"Catalog locations shape: {catalog['locs'].shape}")
            print(f"Average sources per image: {catalog['n_sources'].float().mean():.2f}")
            print(f"Max sources in any image: {catalog['n_sources'].max().item()}")
            
            # Calculate total file sizes
            image_file = f"{save_folder}/images_{config['setting']}.pt"
            catalog_file = f"{save_folder}/catalog_{config['setting']}.pt"
            
            if os.path.exists(image_file) and os.path.exists(catalog_file):
                image_size_mb = os.path.getsize(image_file) / (1024**2)
                catalog_size_mb = os.path.getsize(catalog_file) / (1024**2)
                total_size_mb = image_size_mb + catalog_size_mb
                
                print(f"\n=== FILE SIZES ===")
                print(f"Images file: {image_size_mb:.2f} MB")
                print(f"Catalog file: {catalog_size_mb:.2f} MB")
                print(f"Total: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
            
        except Exception as e:
            print(f"Error verifying final results: {e}")
    
    end_time = time.time()
    end_datetime = datetime.now()
    print(f"\n=== SIMULATION COMPLETED ===")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"=" * 50)

if __name__ == "__main__":
    main()