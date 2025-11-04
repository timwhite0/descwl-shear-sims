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
import anacal 
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ['CATSIM_DIR'] = '/scratch/regier_root/regier0/taodingr/descwl-shear-sims/catsim' 
save_folder = f"/scratch/regier_root/regier0/taodingr/descwl-shear-sims/generated_output"
total, used, free = shutil.disk_usage(save_folder)
print(f"Disk space for {save_folder}:")
print(f"  Total: {total // (2**30)} GiB")
print(f"  Used:  {used // (2**30)} GiB")
print(f"  Free:  {free // (2**30)} GiB")
space_available = free // (2**30)

def create_bright_star_catalog(star_catalog, mag_threshold=18.0, band='r', 
                               radius_scale=5.0, min_radius=5.0, max_radius=50.0):
    """
    FIXED: Handles both regular 2D arrays and structured arrays
    """
    if star_catalog is None or len(star_catalog) == 0:
        return None
    
    try:
        star_data = star_catalog._star_cat
        current_indices = star_catalog.indices
        
        # Get magnitudes
        mag_column = f'{band}_ab'
        if mag_column not in star_data.dtype.names:
            available_bands = [col.replace('_ab', '') for col in star_data.dtype.names if '_ab' in col]
            if available_bands:
                band = available_bands[0]
                mag_column = f'{band}_ab'
            else:
                print(f"WARNING: No magnitude columns found!")
                return None
        
        magnitudes = star_data[mag_column][current_indices]
        bright_mask = np.isfinite(magnitudes) & (magnitudes < mag_threshold)
        n_bright = np.sum(bright_mask)
        
        if n_bright == 0:
            return None
        
        # FIX: Handle different array formats
        star_positions = star_catalog.shifts_array
        
        if len(star_positions.shape) == 2:
            # Regular 2D array: (n_stars, 2)
            star_positions = star_positions[bright_mask]
            
        elif len(star_positions.shape) == 1:
            # Structured array with named fields
            if star_positions.dtype.names is not None:
                filtered_positions = star_positions[bright_mask]
                
                # Try common field name combinations
                if 'x' in star_positions.dtype.names and 'y' in star_positions.dtype.names:
                    star_positions = np.column_stack([filtered_positions['x'], filtered_positions['y']])
                elif 'ra' in star_positions.dtype.names and 'dec' in star_positions.dtype.names:
                    star_positions = np.column_stack([filtered_positions['ra'], filtered_positions['dec']])
                elif len(star_positions.dtype.names) >= 2:
                    # Use first two fields as x, y
                    field1, field2 = star_positions.dtype.names[:2]
                    star_positions = np.column_stack([filtered_positions[field1], filtered_positions[field2]])
                else:
                    print(f"WARNING: Cannot parse position fields: {star_positions.dtype.names}")
                    return None
            else:
                # 1D flattened array - reshape to (N, 2)
                star_positions = star_positions.reshape(-1, 2)
                star_positions = star_positions[bright_mask]
        else:
            print(f"WARNING: Unexpected shape: {star_positions.shape}")
            return None
        
        bright_mags = magnitudes[bright_mask]
        radii = radius_scale * 10**((mag_threshold - bright_mags) / 2.5)
        radii = np.clip(radii, min_radius, max_radius)
        
        bright_star_catalog = np.zeros(n_bright, dtype=[('x', 'f8'), ('y', 'f8'), ('r', 'f8')])
        bright_star_catalog['x'] = star_positions[:, 0]
        bright_star_catalog['y'] = star_positions[:, 1]
        bright_star_catalog['r'] = radii
        
        return bright_star_catalog
        
    except Exception as e:
        print(f"ERROR in create_bright_star_catalog: {e}")
        import traceback
        traceback.print_exc()
        return None

# Default bad mask planes from xlens
badMaskDefault = [
    "BAD",
    "SAT",
    "CR",
    "NO_DATA",
    "UNMASKEDNAN",
    "CROSSTALK",
    "INTRP",
    "STREAK",
    "VIGNETTED",
    "CLIPPED",
]

def extract_mask_xlens_style(exposure, badMaskPlanes=None):
    """Extract mask array following xlens logic"""
    if badMaskPlanes is None:
        badMaskPlanes = badMaskDefault
    
    # Get bitmask for the specified bad mask planes
    bitv = exposure.mask.getPlaneBitMask(badMaskPlanes)
    
    # Check which pixels have ANY of the bad mask planes set
    mask_from_planes = (exposure.mask.array & bitv) != 0
    
    # Mask pixels with highly negative values (< -6 sigma)
    variance_safe = np.where(
        exposure.variance.array < 0,
        0, 
        exposure.variance.array
    )
    negative_threshold = -6.0 * np.sqrt(variance_safe)
    mask_from_negative = exposure.image.array < negative_threshold
    
    # Combine both masks
    mask_array = (mask_from_planes | mask_from_negative).astype(np.int16)
    
    return mask_array

def extract_masks_dict_xlens_style(sim_data, bands, badMaskPlanes=None):
    """Extract masks for all bands following xlens logic"""
    masks_dict = {}
    for band in bands:
        exposure = sim_data['band_data'][band][0]
        mask_array = extract_mask_xlens_style(exposure, badMaskPlanes)
        masks_dict[band] = mask_array
    return masks_dict

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

def save_batch_to_disk(batch_manager, save_anacal=True):
    if batch_manager.current_batch_images is None:
        print("No batch to save!")
        return False
    
    print(f"Saving batch {batch_manager.current_batch_num}...")
    
    batch_image_file = f"{batch_manager.save_folder}/batch_{batch_manager.current_batch_num}_images.pt"
    batch_catalog_file = f"{batch_manager.save_folder}/batch_{batch_manager.current_batch_num}_catalog.pt"
    batch_anacal_file = f"{batch_manager.save_folder}/batch_{batch_manager.current_batch_num}_anacal.pt"
    
    image_success = safe_save_tensor(batch_manager.current_batch_images, batch_image_file)
    catalog_success = safe_save_tensor(batch_manager.current_batch_catalog, batch_catalog_file)
    
    anacal_success = True
    if save_anacal and batch_manager.current_batch_anacal:
        anacal_success = safe_save_tensor(batch_manager.current_batch_anacal, batch_anacal_file)

        cumulative_anacal_file = f"{batch_manager.save_folder}/anacal_results_checkpoint.pt"
        cumulative_success = safe_save_tensor(batch_manager.anacal_results, cumulative_anacal_file)
        if cumulative_success:
            print(f"Checkpoint: Saved cumulative anacal results through batch {batch_manager.current_batch_num}")
    
    overall_success = image_success and catalog_success and anacal_success
    
    if image_success or catalog_success:
        batch_info = {
            'batch_num': batch_manager.current_batch_num,
            'batch_size': batch_manager.current_batch_size,
            'image_file': batch_image_file if image_success else None,
            'catalog_file': batch_catalog_file if catalog_success else None,
            'anacal_file': batch_anacal_file if anacal_success else None,
            'image_shape': batch_manager.current_batch_images.shape if image_success else None,
            'catalog_shape': batch_manager.current_batch_catalog["locs"].shape if catalog_success else None,
        }
        batch_manager.completed_batches.append(batch_info)
    
    if overall_success:
        print(f"Batch {batch_manager.current_batch_num} saved successfully")
    else:
        print(f"Partial save - Images: {image_success}, Catalog: {catalog_success}, Anacal: {anacal_success}")
    
    print(f"Progress: {batch_manager.total_processed}/{batch_manager.total_images} images")
    
    clear_batch_memory(batch_manager)
    return image_success

def clear_batch_memory(batch_manager):
    if batch_manager.current_batch_images is not None:
        del batch_manager.current_batch_images
        del batch_manager.current_batch_catalog
        del batch_manager.current_batch_anacal
        
        batch_manager.current_batch_images = None
        batch_manager.current_batch_catalog = None
        batch_manager.current_batch_anacal = None
    
    cleanup_memory(aggressive=True)
    print(f"Batch {batch_manager.current_batch_num} cleared from memory")

def combine_multiband_images(images_tensor, exposures_list=None, method='inverse_variance'):
    """
    Combine multi-band images into single image for anacal processing.
    
    Parameters:
    -----------
    images_tensor : torch.Tensor
        Shape: [n_bands, height, width]
    exposures_list : list of exposures or None
        If provided, extract variance from actual exposures for proper weighting
    method : str
        'inverse_variance': Weight by 1/variance (xlens approach)
        'weighted': Weight by typical survey depths (old approach)
        'mean': Simple average across bands
    
    Returns:
    --------
    combined_image : torch.Tensor
        Single image [height, width]
    combined_variance : float
        Combined variance value
    """
    
    if method == 'inverse_variance':
        # xlens approach: use inverse-variance weighting
        if exposures_list is not None:
            # Extract actual variance from exposures
            weights = []
            for exposure in exposures_list:
                variance = exposure.getMaskedImage().variance.array
                # Use mean variance as representative (like xlens does)
                finite_variance = variance[np.isfinite(variance)]
                if finite_variance.size == 0:
                    raise ValueError("Variance plane must contain at least one finite value")
                variance_value = float(np.nanmean(variance))
                if not np.isfinite(variance_value) or variance_value <= 0:
                    raise ValueError(f"Invalid variance: {variance_value}")
                weights.append(1.0 / variance_value)
            
            weights = torch.tensor(weights, dtype=torch.float32)
        else:
            # Fallback: use typical LSST variance estimates
            # Based on 5-sigma depths: g~24.8, r~24.4, i~24.0, z~23.3
            # Higher depth = lower variance = higher weight
            typical_variances = torch.tensor([0.354 * 1.2, 0.354 * 0.85, 0.354 * 0.75, 0.354 * 1.0])
            typical_variances = typical_variances[:images_tensor.shape[0]]
            weights = 1.0 / typical_variances
        
        total_weight = weights.sum()
        normalized_weights = weights / total_weight
        combined = torch.sum(images_tensor * normalized_weights.view(-1, 1, 1), dim=0)
        
        # Calculate combined variance (xlens formula)
        combined_variance = 1.0 / total_weight.item()
    
    elif method == 'mean':
        combined = torch.mean(images_tensor, dim=0)
        combined_variance = 0.354  # Default
    
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    return combined, combined_variance

def combine_multiband_masks(masks_dict, method='union'):
    """
    Combine masks from multiple bands into a single mask.
    
    xlens approach: They don't explicitly combine masks in combine_sim_exposures,
    but use the reference (first) exposure's mask. However, for robustness,
    union makes more sense - flag pixel if bad in ANY band.
    """
    if not masks_dict:
        return None
    
    band_names = list(masks_dict.keys())
    first_mask = masks_dict[band_names[0]]
    
    if method == 'union':
        # Logical OR: flag pixel if bad in ANY band (RECOMMENDED for xlens-style)
        # This is more conservative and safer
        combined_mask = np.zeros_like(first_mask, dtype=np.int16)
        for band, mask in masks_dict.items():
            combined_mask = np.logical_or(combined_mask, mask).astype(np.int16)
    
    elif method == 'first_only':
        # Use only first band's mask (what xlens does implicitly)
        combined_mask = first_mask.copy()
    
    elif method == 'intersection':
        # Logical AND: flag pixel only if bad in ALL bands
        combined_mask = np.ones_like(first_mask, dtype=np.int16)
        for band, mask in masks_dict.items():
            combined_mask = np.logical_and(combined_mask, mask).astype(np.int16)
    
    elif method == 'weighted':
        # Weight by typical survey depths
        weights = {'g': 1.0, 'r': 1.5, 'i': 2.0, 'z': 1.5}
        weighted_sum = np.zeros_like(first_mask, dtype=np.float32)
        weight_total = 0.0
        
        for band, mask in masks_dict.items():
            w = weights.get(band, 1.0)
            weighted_sum += mask.astype(np.float32) * w
            weight_total += w
        
        combined_mask = (weighted_sum / weight_total > 0.5).astype(np.int16)
    
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    return combined_mask

def anacal_multiband_combined(images_tensor, catalog, g1, g2, psf_input, masks_dict=None,
                             combine_method='inverse_variance', mask_combine_method='union', 
                             star_catalog=None, npix=64, sigma_arcsec=0.52,
                             mag_zero=30.0, pixel_scale=0.2, noise_variance=0.354,
                             exposures_list=None):  # NEW PARAMETER
    """
    Process multi-band images by combining them first, then running anacal.
    Now uses xlens-style inverse-variance weighting.
    """
    
    # STEP 1: Combine images with proper inverse-variance weighting
    combined_image, combined_variance = combine_multiband_images(
        images_tensor, 
        exposures_list=exposures_list,  # Pass exposures if available
        method=combine_method
    )
    gal_array = combined_image.numpy()
    
    # STEP 2: Combine masks (xlens uses union implicitly through mask propagation)
    combined_mask = None
    if masks_dict is not None:
        combined_mask = combine_multiband_masks(masks_dict, method=mask_combine_method)

    # STEP 3: Generate noise array with COMBINED variance (not original)
    noise_array = torch.normal(
        mean=torch.zeros(gal_array.shape),
        std=np.sqrt(combined_variance) * torch.ones(gal_array.shape)
    ).numpy()
    
    fpfs_config = anacal.fpfs.FpfsConfig(
        npix=npix,
        sigma_arcsec=sigma_arcsec,
    )
    
    if isinstance(psf_input, np.ndarray):
        # Fixed PSF
        out = anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=psf_input,
            pixel_scale=pixel_scale,
            noise_variance=combined_variance,  # Use COMBINED variance
            noise_array=noise_array,
            mask_array=combined_mask,        
            star_catalog=star_catalog, 
            detection=None,
        )
    else:
        # Variable PSF object
        center_psf = psf_input.draw(gal_array.shape[1]//2, gal_array.shape[0]//2)
        out = anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=center_psf,
            psf_object=psf_input,
            pixel_scale=pixel_scale,
            noise_variance=combined_variance,  # Use COMBINED variance
            noise_array=noise_array,
            mask_array=combined_mask,        
            star_catalog=star_catalog, 
            detection=None,
        )
    
    # Extract results (unchanged)
    e1 = out["fpfs_w"] * out["fpfs_e1"]
    e1g1 = out["fpfs_dw_dg1"] * out["fpfs_e1"] + out["fpfs_w"] * out["fpfs_de1_dg1"]
    e1_sum = np.sum(e1)
    e1g1_sum = np.sum(e1g1)
    
    e2 = out["fpfs_w"] * out["fpfs_e2"]
    e2g2 = out["fpfs_dw_dg2"] * out["fpfs_e2"] + out["fpfs_w"] * out["fpfs_de2_dg2"]
    e2_sum = np.sum(e2)
    e2g2_sum = np.sum(e2g2)
    
    num_detections = len(e1)
    
    return e1_sum, e1g1_sum, e2_sum, e2g2_sum, num_detections, out

class GalSimPsfWrapper(anacal.psf.BasePsf):
    """Wrapper to make GalSim PSF objects compatible with anacal"""
    
    def __init__(self, galsim_psf, pixel_scale, npix=64, is_variable=False):
        self.galsim_psf = galsim_psf
        self.pixel_scale = pixel_scale
        self.npix = npix
        self.is_variable = is_variable
        # ADD: Mock shape attribute so anacal thinks this is NOT a numpy array
        self.shape = None  # This signals to anacal that it's not an array
    
    def draw(self, x, y):
        """Draw PSF at position (x, y)"""
        if self.is_variable:
            # Variable PSF - get position-specific PSF
            pos = galsim.PositionD(x, y)
            psf_at_pos = self.galsim_psf.getPSF(pos)
        else:
            # Fixed PSF - same everywhere
            psf_at_pos = self.galsim_psf
        
        return psf_at_pos.drawImage(
            nx=self.npix,
            ny=self.npix,
            scale=self.pixel_scale,
            method='auto'
        ).array.astype(np.float64)

def anacal_single_image(image, catalog, g1, g2, psf_input, mask_array=None, star_catalog=None,
                        npix=64, sigma_arcsec=0.52, 
                        mag_zero=30.0, pixel_scale=0.2, noise_variance=0.354):
    """
    Process a single-band image with anacal
    
    Parameters:
    -----------
    mask_array : np.ndarray or None
        Pixel mask for the single band
    star_catalog : np.ndarray or None
        Bright star catalog for masking
    """
    fpfs_config = anacal.fpfs.FpfsConfig(
        npix=npix,
        sigma_arcsec=sigma_arcsec,
    )

    # Handle different input shapes
    if len(image.shape) == 3:
        if image.shape[0] == 1:
            gal_array = image[0].numpy()
        else:
            raise ValueError(f"Expected single-band image, got {image.shape[0]} bands")
    elif len(image.shape) == 2:
        gal_array = image.numpy()
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    noise_array = torch.normal(
        mean=torch.zeros(gal_array.shape), 
        std=np.sqrt(noise_variance) * torch.ones(gal_array.shape)
    ).numpy()

    # Handle PSF
    if isinstance(psf_input, np.ndarray):
        out = anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=psf_input,
            pixel_scale=pixel_scale,
            noise_variance=noise_variance,
            noise_array=noise_array,
            #mask_array=None,        
            #star_catalog=None,
            mask_array=mask_array,        
            star_catalog=star_catalog,    
            detection=None,
        )
    else:
        center_psf = psf_input.draw(gal_array.shape[1]//2, gal_array.shape[0]//2)
        out = anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=center_psf,
            psf_object=psf_input,
            pixel_scale=pixel_scale,
            noise_variance=noise_variance,
            noise_array=noise_array,
            #mask_array=None,        
            #star_catalog=None,
            mask_array=mask_array,        
            star_catalog=star_catalog,    
            detection=None,
        )

    # Rest remains the same...
    e1 = out["fpfs_w"] * out["fpfs_e1"]
    e1g1 = out["fpfs_dw_dg1"] * out["fpfs_e1"] + out["fpfs_w"] * out["fpfs_de1_dg1"]
    e1_sum = np.sum(e1)
    e1g1_sum = np.sum(e1g1)

    e2 = out["fpfs_w"] * out["fpfs_e2"]
    e2g2 = out["fpfs_dw_dg2"] * out["fpfs_e2"] + out["fpfs_w"] * out["fpfs_de2_dg2"]
    e2_sum = np.sum(e2)
    e2g2_sum = np.sum(e2g2)

    num_detections = len(e1)

    return e1_sum, e1g1_sum, e2_sum, e2g2_sum, num_detections

def get_psf_image(psf_obj, psf_size=64, pixel_scale=0.2, center_pos=None):
    """
    Extract PSF image array consistent with first code's approach
    
    Parameters:
    -----------
    psf_obj : PSF object
        The PSF object from your simulation
    psf_size : int
        Size of the PSF image (psf_size x psf_size)
    pixel_scale : float
        Pixel scale for drawing the PSF
    center_pos : tuple or None
        Position to evaluate PSF at (not used for fixed PSF)
    """
    
    if hasattr(psf_obj, '_im_cen') and hasattr(psf_obj, '_tot_width'):
        # Variable PSF 
        if center_pos is None:
            center_pos = (psf_obj._im_cen, psf_obj._im_cen)
        
        galsim_pos = galsim.PositionD(center_pos[0], center_pos[1])
        psf_galsim_obj = psf_obj.getPSF(galsim_pos)
        
        psf_image = psf_galsim_obj.drawImage(
            nx=psf_size, 
            ny=psf_size,
            scale=psf_obj._scale,  
            method='auto'
        ).array
        
    else:
        # Fixed PSF - match first code's approach exactly
        psf_image = psf_obj.drawImage(
            scale=pixel_scale,
            nx=psf_size, 
            ny=psf_size
        ).array
    
    # Return without normalization to match first code
    # Convert to float64 to match first code's final format
    return psf_image.astype(np.float64)

def create_psf_for_worker(config, se_dim, rng_seed):
    worker_rng = np.random.RandomState(rng_seed)
    
    if config.get('variable_psf', False):
        return make_ps_psf(rng=worker_rng, dim=se_dim, variation_factor=config['variation_factor'])
    else:
        if config['psf_type'] == "gauss":
            return make_fixed_psf(psf_type=config['psf_type']) 
        elif config['psf_type'] == "moffat":  
            return make_fixed_psf(psf_type=config['psf_type'], psf_fwhm=0.8)

def process_single_image_with_anacal(args):
    """Generate image and run anacal in the same worker process"""
    (iter_idx, config, g1_val, g2_val, rng_state, psf_config, layout, shifts, 
     star_config, generate_star, star_setting) = args
    
    rng = np.random.RandomState(config['seed'] + iter_idx)
    
    # Create PSF inside the worker process
    se_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
    psf = create_psf_for_worker(config, se_dim, config['seed'] + iter_idx)
    
    # Generate image
    each_image, positions_tensor, M, magnitude, masks_dict, star_catalog_obj = Generate_single_img_catalog(
        rng, config['mag'], config['hlr'], psf, config['morph'], 
        config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
        config['sep'], g1_val, g2_val, config['bands'], config['noise_factor'], 
        config['dither'], config['dither_size'], config['rotate'], 
        config['cosmic_rays'], config['bad_columns'], config['star_bleeds'], 
        star_config, generate_star, star_setting, shifts, config['catalog_type'], 
        config['select_observable'], config['select_lower_limit'], 
        config['select_upper_limit'], config['draw_bright']
    )
    
    # Crop image
    crop_size = 2048
    h_center, w_center = each_image.shape[1] // 2, each_image.shape[2] // 2
    half_crop = crop_size // 2
    multiband_image = each_image[:, 
                             h_center - half_crop:h_center + half_crop,
                             w_center - half_crop:w_center + half_crop]

    cropped_masks = {}
    for band, mask in masks_dict.items():
        cropped_masks[band] = mask[h_center - half_crop:h_center + half_crop,
                                   w_center - half_crop:w_center + half_crop]

    # Bright star catalog creation (keep this for multi-band)
    bright_star_catalog = None
    if star_catalog_obj is not None:
        star_mag_threshold = config.get('star_catalog', {}).get('mag_threshold', 18.0)
        star_band = config.get('star_catalog', {}).get('band', 'r')
        star_radius_scale = config.get('star_catalog', {}).get('radius_scale', 5.0)
        star_min_radius = config.get('star_catalog', {}).get('min_radius', 5.0)
        star_max_radius = config.get('star_catalog', {}).get('max_radius', 50.0)
        
        bright_star_catalog = create_bright_star_catalog(
            star_catalog_obj,
            mag_threshold=star_mag_threshold,
            band=star_band,
            radius_scale=star_radius_scale,
            min_radius=star_min_radius,
            max_radius=star_max_radius
        )
        
        if bright_star_catalog is not None:
            bright_star_catalog['x'] -= (w_center - half_crop)
            bright_star_catalog['y'] -= (h_center - half_crop)
            
            in_bounds = (
                (bright_star_catalog['x'] >= 0) & 
                (bright_star_catalog['x'] < crop_size) &
                (bright_star_catalog['y'] >= 0) & 
                (bright_star_catalog['y'] < crop_size)
            )
            bright_star_catalog = bright_star_catalog[in_bounds]
            
            if len(bright_star_catalog) == 0:
                bright_star_catalog = None
    
    # Prepare PSF for anacal (INSIDE WORKER)
    if config.get('variable_psf', False):
        psf_for_anacal = GalSimPsfWrapper(
            psf, 
            pixel_scale=config['pixel_scale'],
            npix=config.get('psf_size', 64),
            is_variable=True
        )
    else:
        psf_size = config.get('psf_size', 64)
        psf_for_anacal = get_psf_image(psf, psf_size=psf_size, pixel_scale=config['pixel_scale'])
    
    # ===== NEW: CONDITIONAL ANACAL BASED ON BAND COUNT =====
    n_bands = len(config['bands'])
    
    try:
        if n_bands == 1:
            # SINGLE-BAND CASE: Use simple anacal_single_image
            band_name = config['bands'][0]  # e.g., 'i'
            single_band_mask = cropped_masks.get(band_name, None)

            print(f"Image {iter_idx}: Running SINGLE-BAND anacal for band '{config['bands'][0]}'")
            e1_sum, e1g1_sum, e2_sum, e2g2_sum, num_detections = \
                anacal_single_image(
                    multiband_image,  # Will be [1, H, W] for single band
                    positions_tensor, 
                    g1_val, 
                    g2_val, 
                    psf_for_anacal,
                    mask_array=single_band_mask,         
                    star_catalog=bright_star_catalog, 
                    npix=config.get('psf_size', 64),
                    sigma_arcsec=config.get('sigma_arcsec', 0.52),
                    pixel_scale=config['pixel_scale'],
                    noise_variance=config.get('noise_variance', 0.354)
                )
        else:
            # MULTI-BAND CASE: Use existing multi-band logic
            print(f"Image {iter_idx}: Running MULTI-BAND anacal for bands {config['bands']}")
            e1_sum, e1g1_sum, e2_sum, e2g2_sum, num_detections, _ = \
                anacal_multiband_combined(
                    multiband_image,           # [n_bands, H, W]
                    positions_tensor, 
                    g1_val, 
                    g2_val, 
                    psf_for_anacal,
                    masks_dict=cropped_masks,
                    star_catalog=bright_star_catalog,
                    combine_method=config.get('combine_method', 'inverse_variance'),
                    mask_combine_method=config.get('mask_combine_method', 'first_only'),
                    npix=config.get('psf_size', 64),
                    sigma_arcsec=config.get('sigma_arcsec', 0.52),
                    pixel_scale=config['pixel_scale'],
                    noise_variance=config.get('noise_variance', 0.354)
                )
        
        anacal_results = {
            'e1_sum': float(e1_sum),
            'e1g1_sum': float(e1g1_sum),
            'e2_sum': float(e2_sum),
            'e2g2_sum': float(e2g2_sum),
            'num_detections': int(num_detections)
        }
    except Exception as e:
        print(f"Anacal failed for image {iter_idx}: {e}")
        import traceback
        traceback.print_exc()
        anacal_results = {
            'e1_sum': 0.0,
            'e1g1_sum': 0.0,
            'e2_sum': 0.0,
            'e2g2_sum': 0.0,
            'num_detections': 0
        }
    
    # Return only picklable objects
    return iter_idx, multiband_image, positions_tensor, M, magnitude, anacal_results

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
        
        # Anacal results storage
        self.current_batch_anacal = None
        
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
        
        # Initialize anacal results tensors
        self.anacal_results = {
            'e1_sum': torch.zeros(self.total_images),
            'e1g1_sum': torch.zeros(self.total_images),
            'e2_sum': torch.zeros(self.total_images),
            'e2g2_sum': torch.zeros(self.total_images),
            'num_detections': torch.zeros(self.total_images)
        }

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
        
        # Determine if single or multi-band from image shape
        if len(sample_image.shape) == 3:  # [n_bands, height, width]
            n_bands, height, width = sample_image.shape
        elif len(sample_image.shape) == 2:  # [height, width] - unlikely but handle it
            height, width = sample_image.shape
            n_bands = 1
        else:
            raise ValueError(f"Unexpected image shape: {sample_image.shape}")
        
        print(f"Detected: {n_bands} band(s)")
        print(f"Batch image tensor: [{self.current_batch_size}, {n_bands}, {height}, {width}]")
        
        # Initialize batch image tensor with actual number of bands
        self.current_batch_images = torch.zeros(
            self.current_batch_size, n_bands, height, width, dtype=torch.float32
        )
        
        # Rest remains the same...
        total_sources_estimate = sample_M
        max_sources_per_tile = max(10, total_sources_estimate // (self.n_tiles_per_side ** 2) * 3)
        print(f"Max sources per tile estimate: {max_sources_per_tile}")
    
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

        self.current_batch_anacal = {
            'e1_sum': torch.zeros(self.current_batch_size),
            'e1g1_sum': torch.zeros(self.current_batch_size),
            'e2_sum': torch.zeros(self.current_batch_size),
            'e2g2_sum': torch.zeros(self.current_batch_size),
            'num_detections': torch.zeros(self.current_batch_size)
        }

    def add_image_to_batch(self, global_idx, image, positions, n_sources, g1, g2, psf_image):
        """Add a single image to the current batch with TILED catalog structure and anacal processing"""
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
                print(f"Tile ({tile_row}, {tile_col}) has {n_tile_sources} sources, expanding tensors...")
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
        
        # Process with anacal
        try:
            e1_sum, e1g1_sum, e2_sum, e2g2_sum, num_detections = anacal_single_image(
                image, positions, g1, g2, psf_image
            )
            
            # Store anacal results
            self.current_batch_anacal['e1_sum'][local_idx] = e1_sum
            self.current_batch_anacal['e1g1_sum'][local_idx] = e1g1_sum
            self.current_batch_anacal['e2_sum'][local_idx] = e2_sum
            self.current_batch_anacal['e2g2_sum'][local_idx] = e2g2_sum
            self.current_batch_anacal['num_detections'][local_idx] = num_detections
            
            # Store in global results
            self.anacal_results['e1_sum'][global_idx] = e1_sum
            self.anacal_results['e1g1_sum'][global_idx] = e1g1_sum
            self.anacal_results['e2_sum'][global_idx] = e2_sum
            self.anacal_results['e2g2_sum'][global_idx] = e2g2_sum
            self.anacal_results['num_detections'][global_idx] = num_detections
            
        except Exception as e:
            print(f"Anacal processing failed for image {global_idx}: {e}")

        self.total_processed += 1

    def add_image_to_batch_with_anacal(self, global_idx, image, positions, n_sources, g1, g2, anacal_results):
        """Add a single image to the current batch with pre-computed anacal results"""
        batch_start_idx = (self.current_batch_num - 1) * self.batch_size
        local_idx = global_idx - batch_start_idx
        
        if self.current_batch_images is None:
            self.initialize_batch_tensors(image, n_sources)
        
        # Handle different input shapes - preserve band structure
        if len(image.shape) == 4:  # [1, n_bands, height, width]
            image = image.squeeze(0)  # [n_bands, height, width]
        elif len(image.shape) == 2:  # [height, width] - add channel dimension
            image = image.unsqueeze(0)  # [1, height, width]
        # else: already [n_bands, height, width]
        
        self.current_batch_images[local_idx] = image

        # Catalog handling remains the same
        tile_assignments = self.assign_sources_to_tiles(positions.numpy(), g1, g2)
        max_sources_per_tile = self.current_batch_catalog["locs"].shape[3]
        
        for (tile_row, tile_col), tile_data in tile_assignments.items():
            n_tile_sources = len(tile_data['positions'])
            
            if n_tile_sources > max_sources_per_tile:
                print(f"⚠️  Tile ({tile_row}, {tile_col}) has {n_tile_sources} sources, expanding tensors...")
                self._expand_batch_catalog_tensors(n_tile_sources)
                max_sources_per_tile = n_tile_sources
            
            if n_tile_sources > 0:
                tile_positions = torch.tensor(tile_data['positions'], dtype=torch.float32)
                self.current_batch_catalog["locs"][local_idx, tile_row, tile_col, :n_tile_sources] = tile_positions
                self.current_batch_catalog["n_sources"][local_idx, tile_row, tile_col] = n_tile_sources
                
                g1_tensor = torch.tensor(tile_data['g1_values'], dtype=torch.float32).unsqueeze(-1)
                g2_tensor = torch.tensor(tile_data['g2_values'], dtype=torch.float32).unsqueeze(-1)
                
                self.current_batch_catalog["shear_1"][local_idx, tile_row, tile_col, :n_tile_sources] = g1_tensor
                self.current_batch_catalog["shear_2"][local_idx, tile_row, tile_col, :n_tile_sources] = g2_tensor
        
        # Store pre-computed anacal results
        self.current_batch_anacal['e1_sum'][local_idx] = anacal_results['e1_sum']
        self.current_batch_anacal['e1g1_sum'][local_idx] = anacal_results['e1g1_sum']
        self.current_batch_anacal['e2_sum'][local_idx] = anacal_results['e2_sum']
        self.current_batch_anacal['e2g2_sum'][local_idx] = anacal_results['e2g2_sum']
        self.current_batch_anacal['num_detections'][local_idx] = anacal_results['num_detections']
        
        # Store in global results
        self.anacal_results['e1_sum'][global_idx] = anacal_results['e1_sum']
        self.anacal_results['e1g1_sum'][global_idx] = anacal_results['e1g1_sum']
        self.anacal_results['e2_sum'][global_idx] = anacal_results['e2_sum']
        self.anacal_results['e2g2_sum'][global_idx] = anacal_results['e2g2_sum']
        self.anacal_results['num_detections'][global_idx] = anacal_results['num_detections']
        
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
    star_catalog_obj = None
    if generate_star:
        star_catalog_obj = make_star_catalog(
            rng=rng,  
            coadd_dim=coadd_dim,
            buff=buff,
            pixel_scale=pixel_scale,
            star_config=star_config
        )
        
        # Apply filtering if configured
        star_filter = star_setting.get('star_filter', False)
        if star_filter:
            star_catalog_obj = filter_bright_stars_production(
                star_catalog_obj, 
                mag_threshold=star_setting.get('star_filter_mag', 18), 
                band=star_setting.get('star_filter_band', 'r'), 
                verbose=False  # Set to False to reduce log spam
            )

    if generate_star:
        # generate galaxies and stars
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            star_catalog=star_catalog_obj,
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

    # DESCWL mask extraction (matches xlens as closely as possible)
    masks_dict = {}
    for band in bands:
        exposure = sim_data['band_data'][band][0]
        
        # Get DESCWL's actual mask planes
        try:
            descwl_planes = ["CR", "BAD"]
            bitv = exposure.mask.getPlaneBitMask(descwl_planes)
            mask_from_descwl = (exposure.mask.array & bitv) != 0
        except Exception:
            mask_from_descwl = exposure.mask.array != 0
        
        # Add xlens-style negative pixel detection
        try:
            variance_safe = np.where(exposure.variance.array < 0, 0, exposure.variance.array)
            mask_from_negative = exposure.image.array < (-6.0 * np.sqrt(variance_safe))
            mask_array = (mask_from_descwl | mask_from_negative).astype(np.int16)
        except Exception:
            mask_array = mask_from_descwl.astype(np.int16)
        
        masks_dict[band] = mask_array
    
    return image_tensor, positions_tensor, M, magnitude, masks_dict, star_catalog_obj

def Generate_img_catalog_batched_streaming_multiprocessing(config, n_workers=None):
    """
    Generate images using streaming batch processing with multiprocessing only
    """
    num_data = config['num_data']
    
    # Get memory management settings from config with safe defaults
    memory_config = config.get('memory_management', {})
    batch_size = memory_config.get('batch_size', config.get('batch_size', 50))
    aggressive_gc = memory_config.get('aggressive_gc', True)
    monitor_memory = memory_config.get('monitor_memory', False)
    
    save_folder = f"/scratch/regier_root/regier0/taodingr/descwl-shear-sims/generated_output/{config['setting']}"
    
    print(f"  MULTIPROCESSING-ONLY GENERATION")
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
        
        # MULTIPROCESSING ONLY
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
                future_to_idx = {executor.submit(process_single_image_with_anacal, args): args[0] for args in args_list}
                
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_idx),
                    total=len(batch_indices),
                    desc=f"Batch {batch_num} (MP)",
                    unit="img"
                ):
                    iter_idx, multiband_image, positions_tensor, M, magnitude, anacal_results = future.result()
                    batch_manager.add_image_to_batch_with_anacal(
                        iter_idx, multiband_image, positions_tensor, M, g1[iter_idx], g2[iter_idx], anacal_results)
                    
                    if aggressive_gc and iter_idx % 5 == 0:
                        cleanup_memory(aggressive=True)
            
            print(f"Multiprocessing completed for batch {batch_num}")
        
        except Exception as e:
            print(f"Multiprocessing failed for batch {batch_num}: {e}")
            print("Stopping due to multiprocessing failure")
            break
        
        # Save current batch to disk and clear from memory
        batch_save_success = save_batch_to_disk(batch_manager, save_anacal=True)
        
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
    print(f"Batch files pattern: batch_X_images.pt, batch_X_catalog.pt, batch_X_anacal.pt")
    
    # Save final anacal results
    if successful_batches > 0:
        anacal_save_path = f"{save_folder}/anacal_results_final.pt"
        print(f"Saving final anacal results to {anacal_save_path}")
        safe_save_tensor(batch_manager.anacal_results, anacal_save_path)
    
    return {
        'successful_batches': successful_batches,
        'total_batches': total_batches,
        'save_folder': save_folder,
        'setting': config['setting'],
        'anacal_results': batch_manager.anacal_results
    }

@hydra.main(version_base=None, config_path="/scratch/regier_root/regier0/taodingr/descwl-shear-sims/image_generation", config_name="Anacal_config")
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    start_datetime = datetime.now()
    # Convert config
    config = OmegaConf.to_container(cfg, resolve=True)

    # Specify your directory
    config_save_dir = "/scratch/regier_root/regier0/taodingr/descwl-shear-sims/config_snapshots/Anacal"
    os.makedirs(config_save_dir, exist_ok=True)
    
    # Save config snapshot
    setting_name = config['setting']
    timestamp = start_datetime.strftime('%Y%m%d_%H%M%S')
    config_snapshot_path = os.path.join(config_save_dir, f"config_{setting_name}_{timestamp}.yaml")
    
    with open(config_snapshot_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    print(f"Config snapshot saved to: {config_snapshot_path}")
    
    print(f"=== LARGE TENSOR SIMULATION STARTED ===")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 50)
    with open('Anacal_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    n_workers = config.get('n_workers', None)
    
    print(f"Processing mode: Multiprocessing Only")

    magnitudes = Generate_img_catalog_batched_streaming_multiprocessing(
        config, 
        n_workers=n_workers
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