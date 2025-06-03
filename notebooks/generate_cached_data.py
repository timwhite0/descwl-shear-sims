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

os.environ['CATSIM_DIR'] = '/data/scratch/taodingr/lsst_stack/catsim' 

def Generate_single_img_catalog(
    ntrial, rng, mag, hlr, psf, morph, pixel_scale, layout, coadd_dim, buff, sep, g1, g2, bands, 
    noise_factor, dither, dither_size, rotate, cosmic_rays, bad_columns, star_bleeds, star_catalog, shifts
    ):
    """
    Generate one catalog and image - optimized version
    """
    # Remove unnecessary loop - ntrial should always be 1 for single image generation
    if ntrial != 1:
        print("Warning: ntrial should be 1 for single image generation")
    
    # galaxy catalog; you can make your own
    galaxy_catalog = FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout=layout,
        mag=mag,
        hlr=hlr,
        morph=morph,
        pixel_scale=pixel_scale
    )

    galaxy_catalog.shifts_array = shifts

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
    images = []
    
    for band in bands:
        image_np = sim_data['band_data'][band][0].image.array
        # Ensure contiguous memory layout for multiprocessing compatibility
        image_copy = np.array(image_np, copy=True)
        images.append(torch.from_numpy(image_copy).float())
    
    image_tensor = torch.stack(images, dim=0)
    
    return image_tensor, positions_tensor, M

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
    
    each_image, positions_tensor, M = Generate_single_img_catalog(
        1, rng, config['mag'], config['hlr'], psf, config['morph'], 
        config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
        config['sep'], g1_val, g2_val, config['bands'], config['noise_factor'], 
        config['dither'], config['dither_size'], config['rotate'], 
        config['cosmic_rays'], config['bad_columns'], config['star_bleeds'], 
        star_catalog, shifts
    )
    
    # Optimized cropping
    H, W = each_image.shape[1], each_image.shape[2]
    crop_size = 2048
    
    start_h = (H - crop_size) // 2
    start_w = (W - crop_size) // 2
    
    # Use slice operations for efficiency
    single_image = each_image[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
    
    return single_image, positions_tensor, M

def Generate_img_catalog(config, use_multiprocessing=False, use_threading=False, n_workers=None):
    """
    Generate a number of catalogs and images - optimized version
    
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
    
    # Try multiprocessing first (if enabled)
    if use_multiprocessing and num_data > 1:
        if n_workers is None:
            n_workers = min(mp.cpu_count(), num_data, 8)
        
        print(f"Attempting multiprocessing with {n_workers} workers")
        
        try:
            args_list = []
            rng_state = rng.get_state()
            for iter_idx in range(num_data):
                args_list.append((
                    iter_idx, config, g1[iter_idx], g2[iter_idx], 
                    rng_state, psf, layout, shifts, star_catalog
                ))
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(tqdm(
                    executor.map(process_single_image, args_list),
                    total=num_data,
                    desc="Generating images (multiprocessing)"
                ))
            
            # Unpack multiprocessing results
            images_list = []
            positions = []
            n_sources = []
            
            for single_image, positions_tensor, M in results:
                if single_image.shape[1] != 2048:
                    print("The output dimension is not 2048, check coadd_dim and rotate")
                    break
                images_list.append(single_image)
                positions.append(positions_tensor)
                n_sources.append(M)
                
        except Exception as e:
            print(f"Multiprocessing failed: {e}")
            print("Falling back to threading...")
            use_threading = True
            use_multiprocessing = False
    
    if use_threading and num_data > 1 and not use_multiprocessing:
        # Threading approach - good compromise
        if n_workers is None:
            n_workers = min(4, num_data)  # Conservative for threading
        
        print(f"Using threading with {n_workers} workers")
        
        # For threading, we need to be careful about shared objects
        # Create separate instances for each thread to avoid conflicts
        images_list = []
        positions = []
        n_sources = []
        
        def process_with_threading(iter_idx):
            # Create a new RNG for this thread
            thread_rng = np.random.RandomState(config['seed'] + iter_idx * 1000)
            
            # Create thread-local copies of shared objects to avoid conflicts
            thread_layout = Layout(
                layout_name=config['layout_name'],
                coadd_dim=config['coadd_dim'],
                pixel_scale=config['pixel_scale'],
                buff=config['buff']
            )
            
            # Create thread-local PSF
            if config['psf_type'] == "gauss":
                thread_psf = make_fixed_psf(psf_type=config['psf_type']) 
            elif config['psf_type'] == "moffat":  
                thread_psf = make_fixed_psf(psf_type=config['psf_type'], psf_fwhm=0.8)
            
            # Get shifts for this thread
            thread_shifts = thread_layout.get_shifts(thread_rng, density=config['density'])
            
            # Create thread-local star catalog if needed
            thread_star_catalog = None
            if config['star_catalog'] is not None:
                thread_star_catalog = make_star_catalog(
                    rng=thread_rng,
                    coadd_dim=config['coadd_dim'],
                    buff=config['buff'],
                    pixel_scale=config['pixel_scale'],
                    star_config=config['star_config']
                )
            
            each_image, positions_tensor, M = Generate_single_img_catalog(
                1, thread_rng, config['mag'], config['hlr'], thread_psf, config['morph'], 
                config['pixel_scale'], thread_layout, config['coadd_dim'], config['buff'], 
                config['sep'], g1[iter_idx], g2[iter_idx], config['bands'], 
                config['noise_factor'], config['dither'], config['dither_size'], 
                config['rotate'], config['cosmic_rays'], config['bad_columns'], 
                config['star_bleeds'], thread_star_catalog, thread_shifts
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
                executor.map(process_with_threading, range(num_data)),
                total=num_data,
                desc="Generating images (threading)"
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
    
    elif use_multiprocessing and num_data > 1:
        # Unpack multiprocessing results
        images_list = []
        positions = []
        n_sources = []
        
        for single_image, positions_tensor, M in results:
            if single_image.shape[1] != 2048:
                print("The output dimension is not 2048, check coadd_dim and rotate")
                break
            images_list.append(single_image)
            positions.append(positions_tensor)
            n_sources.append(M)
    
    else:
        # Sequential processing (original approach but optimized)
        images_list = []
        positions = []
        n_sources = []
        
        for iter_idx in tqdm(range(num_data)):
            each_image, positions_tensor, M = Generate_single_img_catalog(
                1, rng, config['mag'], config['hlr'], psf, config['morph'], 
                config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
                config['sep'], g1[iter_idx], g2[iter_idx], config['bands'], 
                config['noise_factor'], config['dither'], config['dither_size'], 
                config['rotate'], config['cosmic_rays'], config['bad_columns'], 
                config['star_bleeds'], star_catalog, shifts
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

    return images, plocs, g1_tensor, g2_tensor, n_sources_tensor, M

def save_img_catalog(N, images, plocs, n_sources, M, g1, g2, setting=None):
    """
    Save image AND catalog - optimized version
    """
    M = plocs.shape[1] 
    
    # More efficient tensor expansion
    shear_1 = g1.unsqueeze(1).expand(-1, M, -1)
    shear_2 = g2.unsqueeze(1).expand(-1, M, -1)

    catalog_dict = {
        "plocs": plocs,
        "n_sources": n_sources,
        "shear_1": shear_1,
        "shear_2": shear_2,
    }
    
    save_folder = f"/data/scratch/taodingr/weak_lensing/descwl/{setting}"
    os.makedirs(save_folder, exist_ok=True)  # More pythonic

    # Save with optimized settings
    torch.save(images, f"{save_folder}/images_{setting}.pt")
    torch.save(catalog_dict, f"{save_folder}/catalog_{setting}.pt")

    print(f"Image saved to: {save_folder}/images_{setting}.pt")
    print(f"Catalog saved to: {save_folder}/catalog_{setting}.pt")

def main():
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
    
    images, plocs, g1, g2, n_sources, M = Generate_img_catalog(
        config, use_multiprocessing=use_multiprocessing, 
        use_threading=use_threading, n_workers=n_workers
    )

    dim_w, dim_h = images.shape[2], images.shape[3]
    print(f"Generated {config['num_data']} images")
    print(f"Image dimension: {dim_w} x {dim_h}")
    print("Saving the images and catalogs ...")
    save_img_catalog(config['num_data'], images, plocs, n_sources, M, g1, g2, setting=config['setting'])

if __name__ == "__main__":
    main()