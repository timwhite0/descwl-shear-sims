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

os.environ['CATSIM_DIR'] = '/data/scratch/taodingr/lsst_stack/catsim' 

import shutil

save_folder = f"/nfs/turbo/lsa-regier/scratch/taodingr/descwl"
total, used, free = shutil.disk_usage(save_folder)
print(f"Disk space for {save_folder}:")
print(f"  Total: {total // (2**30)} GiB")
print(f"  Used:  {used // (2**30)} GiB")
print(f"  Free:  {free // (2**30)} GiB")
space_available = free // (2**30)

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
        
        # Show expected speedup based on test results
        '''
        expected_speedups = {
            15.0: "~30% faster",
            16.0: "~50% faster", 
            18.0: "~92% faster"
        }
        expected = expected_speedups.get(mag_threshold, "speedup varies")
        print(f"Expected performance improvement: {expected}")
        '''
    
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

def check_storage_requirements(first_image, first_catalog_dict, num_total_images, save_folder):
    """
    Estimate total storage requirements based on first image and catalog
    Returns True if enough space, False otherwise
    """
    # Estimate size of one image
    image_size_mb = estimate_tensor_size_mb(first_image)
    
    # Estimate size of one catalog (sum all tensors in catalog_dict)
    catalog_size_mb = 0
    for key, tensor in first_catalog_dict.items():
        catalog_size_mb += estimate_tensor_size_mb(tensor)
    
    # Total size for one complete dataset (image + catalog)
    single_dataset_mb = image_size_mb + catalog_size_mb
    
    # Estimate total size for all images
    total_estimated_mb = single_dataset_mb * num_total_images
    total_estimated_gb = total_estimated_mb / 1024
    
    print(f"\n--- Storage Estimation ---")
    print(f"Single image size: {image_size_mb:.2f} MB")
    print(f"Single catalog size: {catalog_size_mb:.2f} MB")
    print(f"Single dataset size: {single_dataset_mb:.2f} MB")
    print(f"Estimated total size for {num_total_images} images: {total_estimated_gb:.2f} GB")
    
    # Check available disk space
    total, used, free = shutil.disk_usage(save_folder)
    free_gb = free / (1024**3)
    
    print(f"Available disk space: {free_gb:.2f} GB")
    
    # Add 10% safety margin
    safety_margin = 1.1
    required_space_gb = total_estimated_gb * safety_margin
    
    if required_space_gb > free_gb:
        print(f"❌ WARNING: Not enough disk space!")
        print(f"Required space (with 10% margin): {required_space_gb:.2f} GB")
        print(f"Available space: {free_gb:.2f} GB")
        print(f"Shortfall: {required_space_gb - free_gb:.2f} GB")
        return False
    else:
        print(f"✅ Sufficient disk space available")
        print(f"Required space (with 10% margin): {required_space_gb:.2f} GB")
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
            star_bleeds=star_bleeds
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

def pad_to_max_optimized(positions_list):
    """Optimized padding function using vectorized operations"""
    if not positions_list:
        return torch.empty(0)
    
    max_sources = max(p.shape[0] for p in positions_list)
    
    # Pre-allocate the result tensor
    n_images = len(positions_list)
    n_dims = positions_list[0].shape[1] if positions_list else 2
    result = torch.full((n_images, max_sources, n_dims), float('nan'))
    
    # Fill in the actual positions
    for i, p in enumerate(positions_list):
        if p.shape[0] > 0:
            result[i, :p.shape[0]] = p
    
    return result

def process_single_image(args):
    """Function to process a single image - for multiprocessing"""
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
    
    return single_image, positions_tensor, M, magnitude

def Generate_img_catalog(config, use_multiprocessing=False, use_threading=False, n_workers=None):
    """
    Generate a number of catalogs and images - optimized version with storage check
    
    Args:
        use_multiprocessing: Use ProcessPoolExecutor (fastest but may have serialization issues)
        use_threading: Use ThreadPoolExecutor (good compromise, but may have thread safety issues)
        If both are False, uses optimized sequential processing (recommended for stability)
    """
    num_data = config['num_data']
    rng = np.random.RandomState(config['seed'])
    input_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
    print(f"The input image dimension is {input_dim}")
    
    # Pre-generate all shear values at once
    if config['shear_setting'] == "const":
        const = 0.05
        g1 = np.full(num_data, const, dtype=np.float32)
        g2 = np.full(num_data, const, dtype=np.float32)
    elif config['shear_setting'] == "vary":
        # Draw from Gaussian and clip in one go
        g1 = np.clip(np.random.normal(0.0, 0.015, num_data), -0.05, 0.05).astype(np.float32)
        g2 = np.clip(np.random.normal(0.0, 0.015, num_data), -0.05, 0.05).astype(np.float32)
    
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

    # Generate first image to estimate storage requirements
    print("Generating first image to estimate storage requirements...")
    first_image, first_positions, first_M, first_magnitude = Generate_single_img_catalog(
        1, rng, config['mag'], config['hlr'], psf, config['morph'], 
        config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
        config['sep'], g1[0], g2[0], config['bands'], config['noise_factor'], 
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
    first_plocs = first_positions.unsqueeze(0)  # Add batch dimension
    first_g1_tensor = torch.tensor([g1[0]]).float().unsqueeze(1)
    first_g2_tensor = torch.tensor([g2[0]]).float().unsqueeze(1)
    first_n_sources = torch.tensor([first_M], dtype=torch.long)
    
    # For catalog estimation, we need to simulate the full catalog structure
    max_sources_estimate = first_M  # This might be conservative, but safer
    shear_1_estimate = first_g1_tensor.unsqueeze(1).expand(-1, max_sources_estimate, -1)
    shear_2_estimate = first_g2_tensor.unsqueeze(1).expand(-1, max_sources_estimate, -1)
    
    first_catalog_dict = {
        "plocs": first_plocs,
        "n_sources": first_n_sources,
        "shear_1": shear_1_estimate,
        "shear_2": shear_2_estimate,
    }
    
    # Check storage requirements
    save_folder = f"/nfs/turbo/lsa-regier/scratch/taodingr/descwl/{config['setting']}"
    os.makedirs(save_folder, exist_ok=True)
    
    if not check_storage_requirements(first_image_cropped.unsqueeze(0), first_catalog_dict, num_data, save_folder):
        print("❌ STOPPING: Insufficient disk space for all images!")
        print("Consider:")
        print("1. Reducing num_data in config")
        print("2. Using a different storage location")
        print("3. Freeing up disk space")
        return None, None, None, None, None, None
    
    # If we reach here, we have enough space - continue with generation
    print("✅ Sufficient storage available. Continuing with image generation...")
    
    # Initialize lists with first image
    images_list = [first_image_cropped]
    positions = [first_positions]
    n_sources = [first_M]
    magnitudes = [first_magnitude]
    
    # Continue generating remaining images
    remaining_images = num_data - 1
    
    if remaining_images > 0:
        # Try multiprocessing first (if enabled)
        if use_multiprocessing and remaining_images > 0:
            if n_workers is None:
                n_workers = min(mp.cpu_count(), remaining_images, 8)
            
            print(f"Attempting multiprocessing with {n_workers} workers for remaining {remaining_images} images")
            
            try:
                args_list = []
                rng_state = rng.get_state()
                for iter_idx in range(1, num_data):  # Start from 1 since we already have first image
                    args_list.append((
                        iter_idx, config, g1[iter_idx], g2[iter_idx], 
                        rng_state, psf, layout, shifts, star_catalog
                    ))
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(tqdm(
                        executor.map(process_single_image, args_list),
                        total=remaining_images,
                        desc="Generating remaining images (multiprocessing)"
                    ))
                
                # Unpack multiprocessing results
                for single_image, positions_tensor, M, magnitude in results:
                    if single_image.shape[1] != 2048:
                        print("The output dimension is not 2048, check coadd_dim and rotate")
                        break
                    images_list.append(single_image)
                    positions.append(positions_tensor)
                    n_sources.append(M)
                    magnitudes.append(magnitude)
                    
            except Exception as e:
                print(f"Multiprocessing failed: {e}")
                print("Falling back to threading...")
                use_threading = True
                use_multiprocessing = False
        
        if use_threading and remaining_images > 0 and not use_multiprocessing:
            # Threading approach - good compromise
            if n_workers is None:
                n_workers = min(4, remaining_images)  # Conservative for threading
            
            print(f"Using threading with {n_workers} workers for remaining {remaining_images} images")
            
            def process_with_threading(iter_idx):
                # Create a new RNG for this thread
                thread_rng = np.random.RandomState(config['seed'] + iter_idx * 1000)
                
                # Create thread-local copies of shared objects to avoid conflicts
                thread_layout = layout
                thread_psf = psf
                
                # Get shifts for this thread
                thread_shifts = shifts if shifts is not None else layout.get_shifts(thread_rng, density=config['density'])
                
                # Reuse the shared star catalog (thread-safe for read operations)
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
                
                # Optimized cropping
                H, W = each_image.shape[1], each_image.shape[2]
                crop_size = 2048
                
                start_h = (H - crop_size) // 2
                start_w = (W - crop_size) // 2
                
                single_image = each_image[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
                
                return iter_idx, single_image, positions_tensor, M
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                thread_results = list(tqdm(
                    executor.map(process_with_threading, range(1, num_data)),  # Start from 1
                    total=remaining_images,
                    desc="Generating remaining images (threading)"
                ))
            
            # Sort results by index to maintain order
            thread_results.sort(key=lambda x: x[0])
            
            for _, single_image, positions_tensor, M in thread_results:
                if single_image.shape[1] != 2048:
                    print("The output dimension is not 2048, check coadd_dim and rotate")
                    break
                images_list.append(single_image)
                positions.append(positions_tensor)
                n_sources.append(M)
        
        elif not use_multiprocessing and not use_threading and remaining_images > 0:
            # Sequential processing for remaining images
            for iter_idx in tqdm(range(1, num_data), desc="Generating remaining images (sequential)"):
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
                images_list.append(single_image)
                positions.append(positions_tensor)
                n_sources.append(M)
    
    # Optimized tensor operations
    images = torch.stack(images_list, dim=0)
    plocs = pad_to_max_optimized(positions)
    
    # Create tensors more efficiently
    g1_tensor = torch.from_numpy(g1).float().unsqueeze(1)
    g2_tensor = torch.from_numpy(g2).float().unsqueeze(1)
    n_sources_tensor = torch.tensor(n_sources, dtype=torch.long)
    M = max(n_sources) if n_sources else 0

    return images, plocs, g1_tensor, g2_tensor, n_sources_tensor, M, magnitudes

def save_img_catalog(N, images, plocs, n_sources, M, g1, g2, n_tiles_h, n_tiles_w, setting=None):
    """
    Save image AND catalog - optimized version
    """
    M = plocs.shape[1] 
    
    # Pre-allocate tensors
    locs_shape = (N, n_tiles_h, n_tiles_w, M, 2)
    shear_shape = (N, n_tiles_h, n_tiles_w, M, 1)
    n_sources_shape = (N, n_tiles_h, n_tiles_w)
    
    locs = plocs.view(N, 1, 1, M, 2).expand(locs_shape)
    shear_1 = g1.view(N, 1, 1, 1, 1).expand(shear_shape)
    shear_2 = g2.view(N, 1, 1, 1, 1).expand(shear_shape)
    n_sources_reshaped = n_sources.view(N, 1, 1).expand(n_sources_shape)

    catalog_dict = {
        "locs": locs,
        "n_sources": n_sources_reshaped,
        "shear_1": shear_1,
        "shear_2": shear_2,
    }
    
    save_folder = f"/nfs/turbo/lsa-regier/scratch/taodingr/descwl/{setting}"
    os.makedirs(save_folder, exist_ok=True)  

    # Save with optimized settings
    torch.save(images, f"{save_folder}/images_{setting}.pt", 
               _use_new_zipfile_serialization=True)  # Faster serialization
    torch.save(catalog_dict, f"{save_folder}/catalog_{setting}.pt",
               _use_new_zipfile_serialization=True)

    print(f"Image saved to: {save_folder}/images_{setting}.pt")
    print(f"Catalog saved to: {save_folder}/catalog_{setting}.pt")

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


def main():
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"=== SIMULATION STARTED ===")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 50)
    with open('sim_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

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
    
    # apply_speedup_patch()
    result = Generate_img_catalog(
        config, use_multiprocessing=use_multiprocessing, 
        use_threading=use_threading, n_workers=n_workers
    )
    
    # Check if generation was stopped due to insufficient storage
    if result[0] is None:
        print("Image generation stopped due to insufficient disk space.")
        return
    
    images, plocs, g1, g2, n_sources, M, magnitudes = result
    if config['catalog_type'] == 'wldeblend':
        plot_magnitude_distribution(magnitudes, 50, config)

    dim_w, dim_h = images.shape[2], images.shape[3]
    print(f"Generated {images.shape[0]} images")
    print(f"Image dimension: {dim_w} x {dim_h}")
    print("Saving the images and catalogs ...")
    save_img_catalog(images.shape[0], images, plocs, n_sources, M, g1, g2, config['n_tiles_per_side'], config['n_tiles_per_side'], setting=config['setting'])

    end_time = time.time()
    end_datetime = datetime.now()
    print(f"=== SIMULATION COMPLETED ===")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"=" * 50)

if __name__ == "__main__":
    main()