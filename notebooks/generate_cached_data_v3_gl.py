import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from descwl_shear_sims.galaxies import FixedGalaxyCatalog
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.stars import StarCatalog, make_star_catalog
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_shear_sims.psfs import make_ps_psf
from descwl_shear_sims.sim import get_se_dim
from tqdm import tqdm
from descwl_shear_sims.layout.layout import Layout
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
from datetime import datetime
import concurrent.futures
import tempfile
import shutil
import galsim 

os.environ['CATSIM_DIR'] = '/scratch/regier_root/regier0/taodingr/descwl-shear-sims/catsim' 
save_folder = f"/scratch/regier_root/regier0/taodingr/descwl-shear-sims/generated_output"
total, used, free = shutil.disk_usage(save_folder)
print(f"Disk space for {save_folder}:")
print(f"  Total: {total // (2**30)} GiB")
print(f"  Used:  {used // (2**30)} GiB")
print(f"  Free:  {free // (2**30)} GiB")
space_available = free // (2**30)

def cleanup_memory(aggressive=True):
    if aggressive:
        import gc
        gc.collect()

def safe_save_tensor(tensor, filepath, max_retries=3):
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
                print(f"Saved successfully ({size_mb:.1f} MB)")
                return True
            else:
                raise RuntimeError("Final file missing or corrupted")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            if attempt < max_retries - 1:
                time.sleep(1)
    
    print(f"All save attempts failed for {filepath}")
    return False

def save_batch_to_disk(batch_manager, save_psf_params=True):
    if batch_manager.current_batch_images is None:
        print("No batch to save!")
        return False
    
    print(f"Saving batch {batch_manager.current_batch_num}...")
    
    batch_image_file = f"{batch_manager.save_folder}/batch_{batch_manager.current_batch_num}_images.pt"
    batch_catalog_file = f"{batch_manager.save_folder}/batch_{batch_manager.current_batch_num}_catalog.pt"
    batch_psf_file = f"{batch_manager.save_folder}/batch_{batch_manager.current_batch_num}_psf_params.pt"
    
    image_success = safe_save_tensor(batch_manager.current_batch_images, batch_image_file)
    catalog_success = safe_save_tensor(batch_manager.current_batch_catalog, batch_catalog_file)
    
    psf_success = True
    if save_psf_params and batch_manager.current_batch_psf_params:
        psf_success = safe_save_tensor(batch_manager.current_batch_psf_params, batch_psf_file)
    
    overall_success = image_success and catalog_success and psf_success
    
    if image_success or catalog_success:
        batch_info = {
            'batch_num': batch_manager.current_batch_num,
            'batch_size': batch_manager.current_batch_size,
            'image_file': batch_image_file if image_success else None,
            'catalog_file': batch_catalog_file if catalog_success else None,
            'psf_file': batch_psf_file if psf_success else None,
            'image_shape': batch_manager.current_batch_images.shape if image_success else None,
            'catalog_shape': batch_manager.current_batch_catalog["locs"].shape if catalog_success else None,
        }
        batch_manager.completed_batches.append(batch_info)
    
    if overall_success:
        print(f"Batch {batch_manager.current_batch_num} saved successfully")
    else:
        print(f"Partial save - Images: {image_success}, Catalog: {catalog_success}, PSF: {psf_success}")
    
    print(f"Progress: {batch_manager.total_processed}/{batch_manager.total_images} images")
    
    clear_batch_memory(batch_manager)
    return image_success

def clear_batch_memory(batch_manager):
    if batch_manager.current_batch_images is not None:
        del batch_manager.current_batch_images
        del batch_manager.current_batch_catalog
        del batch_manager.current_batch_psf_params
        
        batch_manager.current_batch_images = None
        batch_manager.current_batch_catalog = None
        batch_manager.current_batch_psf_params = None
    
    cleanup_memory(aggressive=True)
    print(f"Batch {batch_manager.current_batch_num} cleared from memory")

def get_psf_param(psf_obj, return_image=True, psf_size=64, center_pos=None):
    """
    Extract PSF parameters AND the PSF image array from different PSF types
    
    Parameters:
    -----------
    psf_obj : PSF object
        The PSF object from your simulation (variable PSF, Gaussian, or Moffat)
    return_image : bool
        Whether to include the actual PSF image array
    psf_size : int
        Size of the PSF image to generate (psf_size x psf_size) - used for fixed PSFs
    center_pos : tuple or None
        Position to evaluate PSF at. If None, uses appropriate center.
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
            
            # Normalize PSF to sum to 1
            psf_image = psf_image / np.sum(psf_image)
            psf_stats['psf_image'] = psf_image.astype(np.float32)
            psf_stats['psf_scale'] = pixel_scale
    
    if return_image:
        psf_stats.update({
            'psf_size': psf_size,
            'psf_center_pos': center_pos
        })
    
    return psf_stats

def create_psf_for_worker(config, se_dim, rng_seed):
    worker_rng = np.random.RandomState(rng_seed)
    
    if config.get('variable_psf', False):
        return make_ps_psf(rng=worker_rng, dim=se_dim, variation_factor=config['variation_factor'])
    else:
        if config['psf_type'] == "gauss":
            return make_fixed_psf(psf_type=config['psf_type']) 
        elif config['psf_type'] == "moffat":  
            return make_fixed_psf(psf_type=config['psf_type'], psf_fwhm=0.8)

def process_single_image(args):
    (iter_idx, config, g1_val, g2_val, rng_state, psf_config, layout, shifts, star_config, generate_star, star_setting) = args
    
    rng = np.random.RandomState()
    rng.set_state(rng_state)
    for _ in range(iter_idx):
        rng.rand()
    
    # Create PSF inside the worker process (CRITICAL FIX)
    se_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
    psf = create_psf_for_worker(config, se_dim, config['seed'] + iter_idx)
    psf_size = config.get('psf_size', 64)
    psf_param = get_psf_param(
        psf, 
        return_image=True, 
        psf_size=psf_size,  # Adjust size as needed
        center_pos=None  # Use image center
    )

    
    each_image, positions_tensor, M, magnitude = Generate_single_img_catalog(
        rng, config['mag'], config['hlr'], psf, config['morph'], 
        config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
        config['sep'], g1_val, g2_val, config['bands'], config['noise_factor'], 
        config['dither'], config['dither_size'], config['rotate'], 
        config['cosmic_rays'], config['bad_columns'], config['star_bleeds'], 
        star_config, generate_star, star_setting, shifts, config['catalog_type'], config['select_observable'],  
        config['select_lower_limit'], config['select_upper_limit'], config['draw_bright']
    )
    
    crop_size = 2048
    h_center, w_center = each_image.shape[1] // 2, each_image.shape[2] // 2
    half_crop = crop_size // 2
    
    single_image = each_image[:, 
                             h_center - half_crop:h_center + half_crop,
                             w_center - half_crop:w_center + half_crop]
    
    return iter_idx, single_image, positions_tensor, M, magnitude, psf_param

class StreamingBatchTensorManager:
    """
    Processes batches one at a time without accumulating in memory
    Saves each batch to disk and combines them at the very end
    """
    def __init__(self, config, save_folder):
        self.config = config
        self.save_folder = save_folder
        self.setting = config['setting']
        
        # Batch configuration
        self.total_images = config['num_data']
        self.batch_size = config.get('batch_size', 200)
        
        # Tiling configuration - NEW
        self.n_tiles_per_side = config.get('n_tiles_per_side', 8)  # Default 8x8 tiling
        self.image_size = 2048  # Assuming 2048x2048 images
        self.tile_size = self.image_size // self.n_tiles_per_side
        
        # File paths
        self.image_file = f"{save_folder}/images_{self.setting}.pt"
        self.catalog_file = f"{save_folder}/catalog_{self.setting}.pt"
        
        # Current batch tensors (only current batch in memory)
        self.current_batch_images = None
        self.current_batch_catalog = None
        self.current_batch_num = 0
        self.current_batch_size = 0
        self.current_batch_psf_params = None
        
        # Track completed batches 
        self.completed_batches = []
        
        # Progress tracking
        self.total_processed = 0
        
        # Create save folder
        os.makedirs(save_folder, exist_ok=True)
        
        # NEW: Tiling info
        print(f"  Tiles per side: {self.n_tiles_per_side}")
        print(f"  Tile size: {self.tile_size}x{self.tile_size} pixels")
        print(f"  Total tiles per image: {self.n_tiles_per_side * self.n_tiles_per_side}")

    def start_new_batch(self, batch_start_idx):
        """Start a new batch and clear previous batch from memory"""
        # Clear previous batch completely
        if self.current_batch_images is not None:
            clear_batch_memory(self)
            print(f"Previous batch cleared from memory")
        
        self.current_batch_num = batch_start_idx // self.batch_size + 1
        batch_end_idx = min(batch_start_idx + self.batch_size, self.total_images)
        self.current_batch_size = batch_end_idx - batch_start_idx
        
        print(f"\n=== STARTING BATCH {self.current_batch_num} ===")
        print(f"Processing images {batch_start_idx} to {batch_end_idx-1} ({self.current_batch_size} images)")
        print(f"Memory available: {self._get_available_memory():.1f} GB")
        
    def _get_available_memory(self):
        """Get available memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except:
            return -1

    def assign_sources_to_tiles(self, positions, g1, g2):
        """
        Assign sources to tiles based on their positions
        
        Returns:
            tile_assignments: dict with tile coordinates as keys and source lists as values
        """
        tile_assignments = {}
        
        for source_idx, (x, y) in enumerate(positions):
            # Convert pixel coordinates to tile indices
            tile_x = int(x // self.tile_size)
            tile_y = int(y // self.tile_size)
            
            # Clamp to valid tile range
            tile_x = max(0, min(tile_x, self.n_tiles_per_side - 1))
            tile_y = max(0, min(tile_y, self.n_tiles_per_side - 1))
            
            tile_key = (tile_y, tile_x)  # (row, col) indexing
            
            if tile_key not in tile_assignments:
                tile_assignments[tile_key] = {
                    'positions': [],
                    'g1_values': [],
                    'g2_values': []
                }
            
            local_x = x - (tile_x * self.tile_size)
            local_y = y - (tile_y * self.tile_size)
            
            tile_assignments[tile_key]['positions'].append([local_x, local_y])
            tile_assignments[tile_key]['g1_values'].append(g1)
            tile_assignments[tile_key]['g2_values'].append(g2)
        
        return tile_assignments
        
    def initialize_batch_tensors(self, sample_image, sample_M):
        """Initialize tensors for current batch with TILED structure"""
        print(f"Initializing tiled batch tensors for {self.current_batch_size} images...")
        
        # Image tensor dimensions: [batch_size, channels, height, width] - UNCHANGED
        channels, height, width = sample_image.shape
        print(f"Batch image tensor: [{self.current_batch_size}, {channels}, {height}, {width}]")
        
        # Initialize batch image tensor 
        self.current_batch_images = torch.zeros(self.current_batch_size, channels, height, width, dtype=torch.float32)
        
        # For tiled catalog, estimate max sources per tile
        total_sources_estimate = sample_M
        max_sources_per_tile = max(10, total_sources_estimate // (self.n_tiles_per_side ** 2) * 3)  # 3x safety factor
        print(f"Max sources per tile estimate: {max_sources_per_tile}")
        
        # Tiled catalog tensor structure
        self.current_batch_catalog = {
            "locs": torch.full((self.current_batch_size, self.n_tiles_per_side, self.n_tiles_per_side, max_sources_per_tile, 2), 
                              float('nan'), dtype=torch.float32),
            "n_sources": torch.zeros((self.current_batch_size, self.n_tiles_per_side, self.n_tiles_per_side), 
                                   dtype=torch.long),
            "shear_1": torch.zeros((self.current_batch_size, self.n_tiles_per_side, self.n_tiles_per_side, max_sources_per_tile, 1), 
                                 dtype=torch.float32),
            "shear_2": torch.zeros((self.current_batch_size, self.n_tiles_per_side, self.n_tiles_per_side, max_sources_per_tile, 1), 
                                 dtype=torch.float32),
        }

        # Initialize PSF parameters dictionary
        self.current_batch_psf_params = {}

    def add_image_to_batch(self, global_idx, image, positions, n_sources, g1, g2, psf_param=None):
        """Add a single image to the current batch with TILED catalog structure"""
        # Calculate local index within current batch
        batch_start_idx = (self.current_batch_num - 1) * self.batch_size
        local_idx = global_idx - batch_start_idx
        
        if self.current_batch_images is None:
            self.initialize_batch_tensors(image, n_sources)
        
        # Add image (remove batch dimension if present)
        if len(image.shape) == 4:
            image = image.squeeze(0)
        self.current_batch_images[local_idx] = image

        tile_assignments = self.assign_sources_to_tiles(positions.numpy(), g1, g2)
        max_sources_per_tile = self.current_batch_catalog["locs"].shape[3]
        
        for (tile_row, tile_col), tile_data in tile_assignments.items():
            n_tile_sources = len(tile_data['positions'])
            
            # Check if we need to expand tensors
            if n_tile_sources > max_sources_per_tile:
                print(f"⚠️  Tile ({tile_row}, {tile_col}) has {n_tile_sources} sources, expanding tensors...")
                self._expand_batch_catalog_tensors(n_tile_sources)
                max_sources_per_tile = n_tile_sources
            
            # Fill in the tile data
            if n_tile_sources > 0:
                # Positions (tile-local coordinates)
                tile_positions = torch.tensor(tile_data['positions'], dtype=torch.float32)
                self.current_batch_catalog["locs"][local_idx, tile_row, tile_col, :n_tile_sources] = tile_positions
                
                # Number of sources in this tile
                self.current_batch_catalog["n_sources"][local_idx, tile_row, tile_col] = n_tile_sources
                
                # Shear values
                g1_tensor = torch.tensor(tile_data['g1_values'], dtype=torch.float32).unsqueeze(-1)
                g2_tensor = torch.tensor(tile_data['g2_values'], dtype=torch.float32).unsqueeze(-1)
                
                self.current_batch_catalog["shear_1"][local_idx, tile_row, tile_col, :n_tile_sources] = g1_tensor
                self.current_batch_catalog["shear_2"][local_idx, tile_row, tile_col, :n_tile_sources] = g2_tensor
        
        # Add PSF parameters
        if psf_param is not None:
            self.current_batch_psf_params[local_idx] = psf_param

        self.total_processed += 1

    def _expand_batch_catalog_tensors(self, new_max_sources):
        """Expand batch catalog tensors to accommodate more sources per tile"""
        old_max = self.current_batch_catalog["locs"].shape[3]
        batch_size, n_tiles_y, n_tiles_x = self.current_batch_catalog["locs"].shape[:3]
        
        # Create new tensors with expanded size
        new_locs = torch.full((batch_size, n_tiles_y, n_tiles_x, new_max_sources, 2), 
                             float('nan'), dtype=torch.float32)
        new_shear_1 = torch.zeros((batch_size, n_tiles_y, n_tiles_x, new_max_sources, 1), 
                                dtype=torch.float32)
        new_shear_2 = torch.zeros((batch_size, n_tiles_y, n_tiles_x, new_max_sources, 1), 
                                dtype=torch.float32)
        
        # Copy existing data
        new_locs[:, :, :, :old_max] = self.current_batch_catalog["locs"]
        new_shear_1[:, :, :, :old_max] = self.current_batch_catalog["shear_1"]
        new_shear_2[:, :, :, :old_max] = self.current_batch_catalog["shear_2"]
        
        # Update batch catalog tensors
        self.current_batch_catalog["locs"] = new_locs
        self.current_batch_catalog["shear_1"] = new_shear_1
        self.current_batch_catalog["shear_2"] = new_shear_2
            
def filter_bright_stars_production(star_catalog, mag_threshold=18, band='r', verbose=True):
    """
    Production-ready bright star filter for massive performance gains

    Parameters:
    -----------
    star_catalog : StarCatalog
        Original star catalog from make_star_catalog()
    mag_threshold : float
        Magnitude threshold - stars brighter (lower mag) than this will be removed
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
        layout='random',  
        coadd_dim=600,    
        buff=0,           
        pixel_scale=0.2,  
        density=n_kept,   
    )
    
    # Copy filtered data
    filtered_catalog._star_cat = star_catalog._star_cat
    filtered_catalog.shifts_array = star_catalog.shifts_array[keep_mask]
    filtered_catalog.indices = star_catalog.indices[keep_mask]
    filtered_catalog.density = star_catalog.density * (n_kept / n_original)
    
    if verbose:
        print(f"Filtered catalog created: {len(filtered_catalog)} stars")
        print(f"Expected to prevent {n_removed} expensive draw_bright_star() calls")
    
    return filtered_catalog

def Generate_single_img_catalog(
    rng, mag, hlr, psf, morph, pixel_scale, layout, coadd_dim, buff, sep, g1, g2, bands, 
    noise_factor, dither, dither_size, rotate, cosmic_rays, bad_columns, star_bleeds, star_config, 
    generate_star, star_setting, shifts, catalog_type, select_observable, select_lower_limit, select_upper_limit, draw_bright, 
    ):
    """
    Generate one catalog and image 
    """
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

    # CREATE STAR CATALOG PER IMAGE
    if generate_star:
        star_catalog = make_star_catalog(
            rng=rng,  
            coadd_dim=coadd_dim,
            buff=buff,
            pixel_scale=pixel_scale,
            star_config=star_config
        )
        
        # Apply filtering if configured
        star_filter = star_setting.get('star_filter', False)
        if star_filter:
            star_catalog = filter_bright_stars_production(
                star_catalog, 
                mag_threshold=star_setting.get('star_filter_mag', 18), 
                band=star_setting.get('star_filter_band', 'r'), 
                verbose=False  # Set to False to reduce log spam
            )

    if generate_star:
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
            draw_bright=draw_bright,
        )    
    else:
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
            star_bleeds=star_bleeds,
        )    

    # get the truth (source) information
    truth = sim_data['truth_info']
    image_x_positions = truth['image_x']
    image_y_positions = truth['image_y']

    x_pos_copy = np.array(image_x_positions, copy=True)
    y_pos_copy = np.array(image_y_positions, copy=True)
    
    positions_tensor = torch.stack([
        torch.from_numpy(x_pos_copy),
        torch.from_numpy(y_pos_copy)
    ], dim=1).float()
    
    M = len(image_x_positions)

    n_bands = len(bands)
    first_band = bands[0]
    h, w = sim_data['band_data'][first_band][0].image.array.shape
    
    image_tensor = torch.zeros(n_bands, h, w, dtype=torch.float32)
    
    for i, band in enumerate(bands):
        image_np = sim_data['band_data'][band][0].image.array
        image_tensor[i] = torch.from_numpy(image_np.copy())
    
    return image_tensor, positions_tensor, M, magnitude

def Generate_img_catalog_batched_streaming_fixed(config, use_multiprocessing=False, n_workers=None, stop_on_multiprocessing_fail=True):
    """
    Generate images using streaming batch processing, supports sequential and multiprocessing modes 
    """
    num_data = config['num_data']
    
    # Get memory management settings from config with safe defaults
    memory_config = config.get('memory_management', {})
    batch_size = memory_config.get('batch_size', config.get('batch_size', 50))
    aggressive_gc = memory_config.get('aggressive_gc', True)
    monitor_memory = memory_config.get('monitor_memory', False)
    
    save_folder = f"/scratch/regier_root/regier0/taodingr/descwl-shear-sims/generated_output/{config['setting']}"
    
    print(f"  BATCH-ONLY GENERATION")
    print(f"   Save location: {save_folder}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total images: {num_data}")
    print(f"   Total batches: {(num_data + batch_size - 1) // batch_size}")
    print(f"   Aggressive GC: {aggressive_gc}")
    
    # Initialize streaming batch tensor manager with custom batch size
    config_copy = config.copy()
    config_copy['batch_size'] = batch_size
    batch_manager = StreamingBatchTensorManager(config_copy, save_folder)
    
    # Simple memory monitoring
    def get_memory_status():
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.percent:.1f}% used, {memory.available/1024**3:.1f}GB available"
        except:
            return "Memory info unavailable"
    
    if monitor_memory:
        print(f"Initial memory status: {get_memory_status()}")
    
    # Basic setup
    rng = np.random.RandomState(config['seed'])
    input_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
    print(f"The original input image dimension is {input_dim}")
    
    # Pre-generate all shear values
    if config['shear_setting'] == "const":
        const = 0.05
        g1 = np.full(num_data, const, dtype=np.float32)
        g2 = np.full(num_data, const, dtype=np.float32)
    elif config['shear_setting'] == "vary":
        rng_shear = np.random.RandomState(config['seed'] + 1000)
        g1 = np.clip(rng_shear.normal(0.0, 0.015, num_data), -0.05, 0.05).astype(np.float32)
        g2 = np.clip(rng_shear.normal(0.0, 0.015, num_data), -0.05, 0.05).astype(np.float32)
    
    layout = Layout(
        layout_name=config['layout_name'],
        coadd_dim=config['coadd_dim'],
        pixel_scale=config['pixel_scale'],
        buff=config['buff']
    )

    shifts = layout.get_shifts(rng, density=config['density'])
    
    # Process in batches 
    successful_batches = 0
    total_batches = (num_data + batch_size - 1) // batch_size
    
    for batch_start in range(0, num_data, batch_size):
        batch_end = min(batch_start + batch_size, num_data)
        batch_num = batch_start // batch_size + 1
        
        print(f"\n=== BATCH {batch_num}/{total_batches} ===")
        print(f"Processing images {batch_start} to {batch_end-1}")
        
        # Memory check before starting new batch
        if monitor_memory:
            print(f"Memory before batch: {get_memory_status()}")
        
        # Start new batch (clears previous batch from memory)
        batch_manager.start_new_batch(batch_start)
        
        # Generate images for current batch
        batch_indices = list(range(batch_start, batch_end))
        
        print(f"Generating {len(batch_indices)} images...")
        
        # ===== SEQUENTIAL PROCESSING =====
        if not use_multiprocessing:
            print("Using sequential processing")
            
            # Create PSF once for sequential processing
            se_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
            psf = create_psf_for_worker(config, se_dim, config['seed'])
            
            for iter_idx in tqdm(batch_indices, desc=f"Batch {batch_num}"):
                each_image, positions_tensor, M, magnitude = Generate_single_img_catalog(
                    rng, config['mag'], config['hlr'], psf, config['morph'], 
                    config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
                    config['sep'], g1[iter_idx], g2[iter_idx], config['bands'], 
                    config['noise_factor'], config['dither'], config['dither_size'], 
                    config['rotate'], config['cosmic_rays'], config['bad_columns'], 
                    config['star_bleeds'], config.get('star_config'), config['generate_star'], 
                    config.get('star_setting'), shifts, config['catalog_type'], 
                    config['select_observable'], config['select_lower_limit'], 
                    config['select_upper_limit'], config['draw_bright']
                )
                
                # Crop image
                H, W = each_image.shape[1], each_image.shape[2]
                crop_size = 2048
                start_h = (H - crop_size) // 2
                start_w = (W - crop_size) // 2
                single_image = each_image[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
                
                batch_manager.add_image_to_batch(iter_idx, single_image, positions_tensor, M, g1[iter_idx], g2[iter_idx])
                
                if aggressive_gc and iter_idx % 3 == 0:
                    cleanup_memory(aggressive=True)
        
        # ===== MULTIPROCESSING =====
        else:
            if n_workers is None:
                n_workers = min(mp.cpu_count(), len(batch_indices), 6)
            
            print(f"Using multiprocessing with {n_workers} workers")
            
            try:
                args_list = []
                rng_state = rng.get_state()
                se_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
                
                # Create PSF config instead of PSF object
                psf_config = {
                    'variable_psf': config.get('variable_psf', False),
                    'psf_type': config['psf_type'],
                    'se_dim': se_dim
                }
                
                for iter_idx in batch_indices:
                    args_list.append((
                        iter_idx, config, g1[iter_idx], g2[iter_idx], 
                        rng_state, psf_config, layout, shifts, 
                        config.get('star_config'), config['generate_star'], config.get('star_setting')
                    ))
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    future_to_idx = {executor.submit(process_single_image, args): args[0] for args in args_list}
                    
                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_idx),
                        total=len(batch_indices),
                        desc=f"Batch {batch_num} (MP)",
                        unit="img"
                    ):
                        iter_idx, single_image, positions_tensor, M, magnitude, psf_param = future.result()
                        batch_manager.add_image_to_batch(
                            iter_idx, single_image, positions_tensor, M, g1[iter_idx], g2[iter_idx], psf_param)
                        
                        if aggressive_gc and iter_idx % 5 == 0:
                            cleanup_memory(aggressive=True)
                
                print(f"Multiprocessing completed for batch {batch_num}")
            
            except Exception as e:
                print(f"Multiprocessing failed: {e}")
                if stop_on_multiprocessing_fail:
                    print("Stopping due to multiprocessing failure")
                    break
        
        # Save current batch to disk and clear from memory
        batch_save_success = save_batch_to_disk(batch_manager, config['save_psf_param'])
        
        if batch_save_success:
            successful_batches += 1
            print(f"Batch {batch_num} saved successfully")
        else:
            print(f"Batch {batch_num} save failed")
            break
        
        # Force garbage collection after each batch
        if aggressive_gc:
            cleanup_memory(aggressive=True)
        
        if monitor_memory:
            print(f"Memory after batch: {get_memory_status()}")
    
    # Final summary 
    print(f"\n=== BATCH GENERATION COMPLETED ===")
    print(f"Successfully generated: {successful_batches}/{total_batches} batches")
    print(f"Batch files location: {save_folder}")
    print(f"Batch files pattern: batch_X_images.pt, batch_X_catalog.pt")
    
    return {
        'successful_batches': successful_batches,
        'total_batches': total_batches,
        'save_folder': save_folder,
        'setting': config['setting']
    }

def main():
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"=== LARGE TENSOR SIMULATION STARTED ===")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 50)
    with open('sim_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    use_multiprocessing = config.get('use_multiprocessing', False)  # Default to False due to serialization issues
    n_workers = config.get('n_workers', None)
    
    processing_mode = "Sequential"
    if use_multiprocessing:
        processing_mode = "Multiprocessing"
    
    print(f"Processing mode: {processing_mode}")
    stop_on_multiprocessing_fail = config.get('stop_on_multiprocessing_fail', True)

    magnitudes = Generate_img_catalog_batched_streaming_fixed(
        config, 
        use_multiprocessing=use_multiprocessing, 
        n_workers=n_workers,
        stop_on_multiprocessing_fail=stop_on_multiprocessing_fail 
    )
    
    # Check if generation was stopped due to insufficient storage
    if magnitudes is None:
        print("Image generation stopped due to insufficient disk space.")
        return
    
    end_time = time.time()
    end_datetime = datetime.now()
    print(f"\n=== SIMULATION COMPLETED ===")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"=" * 50)

if __name__ == "__main__":
    main()