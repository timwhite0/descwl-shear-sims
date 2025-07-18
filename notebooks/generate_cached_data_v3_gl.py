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
import tempfile
import shutil

os.environ['CATSIM_DIR'] = '/scratch/regier_root/regier0/taodingr/descwl-shear-sims/catsim' 

import shutil

save_folder = f"/scratch/regier_root/regier0/taodingr/descwl-shear-sims/generated_output"
total, used, free = shutil.disk_usage(save_folder)
print(f"Disk space for {save_folder}:")
print(f"  Total: {total // (2**30)} GiB")
print(f"  Used:  {used // (2**30)} GiB")
print(f"  Free:  {free // (2**30)} GiB")
space_available = free // (2**30)


def create_psf_for_worker(config, se_dim, rng_seed):
    """
    Create PSF inside the worker process to avoid serialization issues
    """
    # Create a new RNG for this worker
    worker_rng = np.random.RandomState(rng_seed)
    
    if config.get('variable_psf', False):
        return make_ps_psf(rng=worker_rng, dim=se_dim, variation_factor=config['variation_factor'])
    else:
        if config['psf_type'] == "gauss":
            return make_fixed_psf(psf_type=config['psf_type']) 
        elif config['psf_type'] == "moffat":  
            return make_fixed_psf(psf_type=config['psf_type'], psf_fwhm=0.8)

def process_single_image_for_accumulation_fixed(args):
    """Function to process a single image - FIXED for PSF serialization"""
    (iter_idx, config, g1_val, g2_val, rng_state, psf_config, layout, shifts, star_catalog) = args
    
    # Create new RNG with the passed state
    rng = np.random.RandomState()
    rng.set_state(rng_state)
    # Advance the RNG state for this iteration to ensure uniqueness
    for _ in range(iter_idx):
        rng.rand()
    
    # Create PSF inside the worker process (CRITICAL FIX)
    se_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
    psf = create_psf_for_worker(config, se_dim, config['seed'] + iter_idx)
    
    each_image, positions_tensor, M, magnitude = Generate_single_img_catalog(
        1, rng, config['mag'], config['hlr'], psf, config['morph'], 
        config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
        config['sep'], g1_val, g2_val, config['bands'], config['noise_factor'], 
        config['dither'], config['dither_size'], config['rotate'], 
        config['cosmic_rays'], config['bad_columns'], config['star_bleeds'], 
        star_catalog, shifts, config['catalog_type'], config['select_observable'], 
        config['select_lower_limit'], config['select_upper_limit'], config['draw_bright']
    )
    
    crop_size = 2048
    h_center, w_center = each_image.shape[1] // 2, each_image.shape[2] // 2
    half_crop = crop_size // 2
    
    single_image = each_image[:, 
                             h_center - half_crop:h_center + half_crop,
                             w_center - half_crop:w_center + half_crop]
    
    return iter_idx, single_image, positions_tensor, M, magnitude

def ultra_robust_atomic_save(tensor, filepath, max_retries=5):
    """
    Ultra-robust save using atomic writes in the SAME directory
    """
    print(f"    Saving tensor to {os.path.basename(filepath)}...")
    
    target_dir = os.path.dirname(filepath)
    target_filename = os.path.basename(filepath)
    
    # Strategy 1: Atomic write in same directory
    for attempt in range(max_retries):
        try:
            print(f"      Attempt {attempt + 1}: Atomic write in same directory")
            
            # Create temporary file in SAME directory for atomic move
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.tmp', 
                prefix=f'{target_filename}.', 
                dir=target_dir
            )
            os.close(temp_fd)  # Close file descriptor, we'll use the path
            
            # Prepare clean tensor
            if hasattr(tensor, 'detach'):
                clean_tensor = tensor.detach().cpu().contiguous()
            else:
                clean_tensor = tensor
            
            # Force sync before write
            os.sync()
            time.sleep(0.1)  # Small delay to let I/O settle
            
            # Save to temporary file with most compatible format
            torch.save(clean_tensor, temp_path, _use_new_zipfile_serialization=False)
            
            # Force sync after write
            os.sync()
            time.sleep(0.1)
            
            # Verify temp file is complete and not corrupted
            temp_size = os.path.getsize(temp_path)
            if temp_size < 1000:  # Less than 1KB is suspicious
                raise RuntimeError(f"Temp file suspiciously small: {temp_size} bytes")
            
            # Quick verification by attempting to load
            try:
                test_load = torch.load(temp_path, weights_only=True, map_location='cpu')
                if hasattr(tensor, 'shape') and hasattr(test_load, 'shape'):
                    if test_load.shape != tensor.shape:
                        raise RuntimeError(f"Shape verification failed: {test_load.shape} vs {tensor.shape}")
                del test_load
            except Exception as e:
                raise RuntimeError(f"Verification load failed: {e}")
            
            # Atomic move (this is the critical part)
            shutil.move(temp_path, filepath)
            
            # Final verification
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                final_size_mb = os.path.getsize(filepath) / (1024**2)
                print(f"      ✅ Atomic save successful ({final_size_mb:.1f} MB)")
                return True
            else:
                raise RuntimeError("Final file missing or too small after atomic move")
                
        except Exception as e:
            print(f"      ❌ Atomic attempt {attempt + 1} failed: {e}")
            
            # Clean up any temp files
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt)  # Exponential backoff
                print(f"      Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    # Strategy 2: Non-ZIP format fallback (still same directory)
    print(f"      Trying legacy format in same directory...")
    try:
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tmp', 
            prefix=f'{target_filename}.legacy.', 
            dir=target_dir
        )
        os.close(temp_fd)
        
        if hasattr(tensor, 'detach'):
            clean_tensor = tensor.detach().cpu().contiguous()
        else:
            clean_tensor = tensor
        
        # Use older, more reliable format
        torch.save(clean_tensor, temp_path, _use_new_zipfile_serialization=False)
        
        # Verify and move
        if os.path.getsize(temp_path) > 1000:
            shutil.move(temp_path, filepath)
            
            if os.path.exists(filepath):
                print(f"      ✅ Legacy format save successful ({os.path.getsize(filepath) / (1024**2):.1f} MB)")
                return True
        
    except Exception as e:
        print(f"      ❌ Legacy format failed: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
    
    # Strategy 3: Uncompressed format (same directory)
    print(f"      Trying uncompressed format in same directory...")
    try:
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tmp', 
            prefix=f'{target_filename}.uncompressed.', 
            dir=target_dir
        )
        os.close(temp_fd)
        
        if hasattr(tensor, 'detach'):
            clean_tensor = tensor.detach().cpu().contiguous()
        else:
            clean_tensor = tensor
        
        # Save without compression
        with open(temp_path, 'wb') as f:
            torch.save(clean_tensor, f)
        
        if os.path.getsize(temp_path) > 1000:
            shutil.move(temp_path, filepath)
            
            if os.path.exists(filepath):
                print(f"      ✅ Uncompressed save successful ({os.path.getsize(filepath) / (1024**2):.1f} MB)")
                return True
        
    except Exception as e:
        print(f"      ❌ Uncompressed format failed: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
    
    print(f"      ❌ All atomic save strategies failed for {filepath}")
    return False

def wait_for_filesystem_stability(directory, max_wait=30):
    """
    Wait for filesystem to become stable before critical operations
    """
    print(f"    Waiting for filesystem stability...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            # Test basic I/O
            test_file = os.path.join(directory, f'stability_test_{int(time.time())}.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            
            # Read it back
            with open(test_file, 'r') as f:
                content = f.read()
            
            os.remove(test_file)
            
            if content == 'test':
                print(f"    ✅ Filesystem stable after {time.time() - start_time:.1f}s")
                return True
                
        except Exception:
            time.sleep(1)
            continue
    
    print(f"    ⚠️  Filesystem still unstable after {max_wait}s")
    return False

def safe_tensor_copy(tensor):
    """
    Create a safe copy of a tensor for saving
    """
    try:
        if hasattr(tensor, 'detach'):
            # PyTorch tensor
            safe_tensor = tensor.detach().cpu().contiguous().clone()
            # Force memory cleanup
            del tensor
            import gc
            gc.collect()
            return safe_tensor
        else:
            # Already a safe tensor or other type
            return tensor
    except Exception as e:
        print(f"      ⚠️  Tensor copy failed: {e}, using original")
        return tensor

def ultra_robust_catalog_save_improved(catalog_dict, filepath):
    """Improved catalog save with atomic writes"""
    print(f"    Saving catalog to {os.path.basename(filepath)}...")
    
    try:
        # Create safe copies of all tensors
        safe_catalog = {}
        for key, tensor in catalog_dict.items():
            print(f"      Processing {key}...")
            safe_catalog[key] = safe_tensor_copy(tensor)
        
        # Use the improved atomic save
        return ultra_robust_atomic_save(safe_catalog, filepath)
        
    except Exception as e:
        print(f"    ❌ Catalog processing failed: {e}")
        return False

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
        self.n_tiles_per_side = config.get('n_tiles_per_side', 4)  # Default 4x4 tiling
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
        
        # Track completed batches (only metadata, not tensors)
        self.completed_batches = []
        
        # Progress tracking
        self.total_processed = 0
        
        # Create save folder
        os.makedirs(save_folder, exist_ok=True)
        
        print(f"Streaming batch processing configuration:")
        print(f"  Total images: {self.total_images}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Number of batches: {(self.total_images + self.batch_size - 1) // self.batch_size}")
        print(f"  Memory strategy: Process one batch at a time (streaming)")
        
        # NEW: Tiling info
        print(f"  Tiles per side: {self.n_tiles_per_side}")
        print(f"  Tile size: {self.tile_size}x{self.tile_size} pixels")
        print(f"  Total tiles per image: {self.n_tiles_per_side * self.n_tiles_per_side}")

    def start_new_batch(self, batch_start_idx):
        """Start a new batch and clear previous batch from memory - UNCHANGED"""
        # Clear previous batch completely
        if self.current_batch_images is not None:
            del self.current_batch_images
            del self.current_batch_catalog
            self.current_batch_images = None
            self.current_batch_catalog = None
            
            # Force garbage collection
            import gc
            gc.collect()
            print(f"🗑️  Previous batch cleared from memory")
        
        self.current_batch_num = batch_start_idx // self.batch_size + 1
        batch_end_idx = min(batch_start_idx + self.batch_size, self.total_images)
        self.current_batch_size = batch_end_idx - batch_start_idx
        
        print(f"\n=== STARTING BATCH {self.current_batch_num} ===")
        print(f"Processing images {batch_start_idx} to {batch_end_idx-1} ({self.current_batch_size} images)")
        print(f"Memory available: {self._get_available_memory():.1f} GB")
        
    def _get_available_memory(self):
        """Get available memory in GB - UNCHANGED"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except:
            return -1

    def assign_sources_to_tiles(self, positions, g1, g2):
        """
        NEW METHOD: Assign sources to tiles based on their positions
        
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
            
            # Convert to tile-local coordinates
            local_x = x - (tile_x * self.tile_size)
            local_y = y - (tile_y * self.tile_size)
            
            tile_assignments[tile_key]['positions'].append([local_x, local_y])
            tile_assignments[tile_key]['g1_values'].append(g1)
            tile_assignments[tile_key]['g2_values'].append(g2)
        
        return tile_assignments
        
    def initialize_batch_tensors(self, sample_image, sample_positions, sample_M, sample_g1, sample_g2):
        """Initialize tensors for current batch with TILED structure - MODIFIED"""
        print(f"Initializing tiled batch tensors for {self.current_batch_size} images...")
        
        # Image tensor dimensions: [batch_size, channels, height, width] - UNCHANGED
        channels, height, width = sample_image.shape
        
        # Calculate memory requirement
        batch_memory_gb = (self.current_batch_size * channels * height * width * 4) / (1024**3)
        print(f"Batch image tensor: [{self.current_batch_size}, {channels}, {height}, {width}]")
        print(f"Estimated batch memory: {batch_memory_gb:.2f} GB")
        
        # Initialize batch image tensor - UNCHANGED
        self.current_batch_images = torch.zeros(self.current_batch_size, channels, height, width, dtype=torch.float32)
        
        # MODIFIED: For tiled catalog, estimate max sources per tile
        total_sources_estimate = sample_M
        max_sources_per_tile = max(10, total_sources_estimate // (self.n_tiles_per_side ** 2) * 3)  # 3x safety factor
        
        catalog_memory_gb = (self.current_batch_size * self.n_tiles_per_side * self.n_tiles_per_side * max_sources_per_tile * 8) / (1024**3)
        print(f"Max sources per tile estimate: {max_sources_per_tile}")
        print(f"Estimated catalog memory: {catalog_memory_gb:.2f} GB")
        
        # COMPLETELY NEW: Tiled catalog tensor structure
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
        
        print(f"✅ Tiled batch tensors initialized successfully!")
        print(f"📊 Catalog shape: {self.current_batch_catalog['locs'].shape}")
        print(f"📊 Total estimated batch memory: {batch_memory_gb + catalog_memory_gb:.2f} GB")

    def add_image_to_batch(self, global_idx, image, positions, n_sources, g1, g2):
        """Add a single image to the current batch with TILED catalog structure - COMPLETELY MODIFIED"""
        # Calculate local index within current batch
        batch_start_idx = (self.current_batch_num - 1) * self.batch_size
        local_idx = global_idx - batch_start_idx
        
        if self.current_batch_images is None:
            self.initialize_batch_tensors(image, positions, n_sources, g1, g2)
        
        # Add image (remove batch dimension if present) - UNCHANGED
        if len(image.shape) == 4:
            image = image.squeeze(0)
        self.current_batch_images[local_idx] = image
        
        # NEW: Assign sources to tiles
        tile_assignments = self.assign_sources_to_tiles(positions.numpy(), g1, g2)
        
        # NEW: Populate tiled catalog
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
        
        self.total_processed += 1

    def _expand_batch_catalog_tensors(self, new_max_sources):
        """NEW METHOD: Expand batch catalog tensors to accommodate more sources per tile"""
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
        
    def save_current_batch_to_disk_improved(self):
        """Save current batch with filesystem stability checks"""
        if self.current_batch_images is None:
            print("❌ No batch to save!")
            return False
            
        print(f"💾 Saving batch {self.current_batch_num} to disk...")
        
        # Wait for filesystem stability
        if not wait_for_filesystem_stability(self.save_folder):
            print("⚠️  Proceeding despite filesystem instability...")
        
        # Create safe copies before saving to prevent corruption
        print(f"  Creating safe tensor copies...")
        safe_images = safe_tensor_copy(self.current_batch_images)
        safe_catalog = {}
        for key, tensor in self.current_batch_catalog.items():
            safe_catalog[key] = safe_tensor_copy(tensor)
    
        # Force memory cleanup and system sync
        import gc
        gc.collect()
        os.sync()  # Force all pending I/O to complete
        time.sleep(0.5)  # Let I/O settle
        
        # Save batch files
        batch_image_file = f"{self.save_folder}/batch_{self.current_batch_num}_images.pt"
        batch_catalog_file = f"{self.save_folder}/batch_{self.current_batch_num}_catalog.pt"
        
        # Save images with improved atomic method
        print(f"  Saving images with improved atomic method...")
        image_save_success = ultra_robust_atomic_save(safe_images, batch_image_file)
        
        # Wait between saves to reduce I/O pressure
        if image_save_success:
            print(f"  Waiting for I/O to settle before catalog save...")
            time.sleep(2)
            os.sync()
    
        # Save catalog with improved atomic method
        print(f"  Saving catalog with improved atomic method...")
        catalog_save_success = ultra_robust_catalog_save_improved(safe_catalog, batch_catalog_file)
        
        # Determine overall success
        overall_success = image_save_success and catalog_save_success
        
        if overall_success:
            print(f"✅ Both images and catalog saved successfully")
        elif image_save_success and not catalog_save_success:
            print(f"⚠️  Images saved but catalog failed - batch partially saved")
            print(f"    You can regenerate just the catalog later if needed")
        elif not image_save_success and catalog_save_success:
            print(f"⚠️  Catalog saved but images failed - removing orphaned catalog")
            try:
                os.remove(batch_catalog_file)
            except:
                pass
        else:
            print(f"❌ Both images and catalog saves failed")
    
        # Record batch metadata (even for partial success)
        if image_save_success or catalog_save_success:
            batch_info = {
                'batch_num': self.current_batch_num,
                'batch_size': self.current_batch_size,
                'image_file': batch_image_file if image_save_success else None,
                'catalog_file': batch_catalog_file if catalog_save_success else None,
                'image_shape': safe_images.shape if image_save_success else None,
                'catalog_shape': safe_catalog["locs"].shape if catalog_save_success else None,
                'partial_save': not overall_success
            }
            self.completed_batches.append(batch_info)
            
            print(f"📁 Files saved:")
            if image_save_success:
                print(f"  ✅ Images: {os.path.basename(batch_image_file)}")
            if catalog_save_success:
                print(f"  ✅ Catalog: {os.path.basename(batch_catalog_file)}")
        
        print(f"📊 Progress: {self.total_processed}/{self.total_images} images")
    
        # Clear current batch from memory immediately
        del self.current_batch_images
        del self.current_batch_catalog
        del safe_images
        del safe_catalog
        self.current_batch_images = None
        self.current_batch_catalog = None
        
        # Force garbage collection and sync
        gc.collect()
        os.sync()
        
        print(f"🗑️  Batch {self.current_batch_num} cleared from memory")
        print(f"💾 Available memory: {self._get_available_memory():.1f} GB")
        
        # Return True if at least images were saved (most important part)
        return image_save_success

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

def Generate_single_img_catalog(
    ntrial, rng, mag, hlr, psf, morph, pixel_scale, layout, coadd_dim, buff, sep, g1, g2, bands, 
    noise_factor, dither, dither_size, rotate, cosmic_rays, bad_columns, star_bleeds, star_catalog, shifts,
    catalog_type, select_observable, select_lower_limit, select_upper_limit, draw_bright
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
            star_bleeds=star_bleeds,
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
            draw_bright=draw_bright,
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
        config['select_lower_limit'], config['select_upper_limit'], config['draw_bright']
    )
    
    crop_size = 2048
    h_center, w_center = each_image.shape[1] // 2, each_image.shape[2] // 2
    half_crop = crop_size // 2
    
    single_image = each_image[:, 
                             h_center - half_crop:h_center + half_crop,
                             w_center - half_crop:w_center + half_crop]
    
    return iter_idx, single_image, positions_tensor, M, magnitude

def Generate_img_catalog_batched_streaming_fixed(config, use_multiprocessing=False, use_threading=False, n_workers=None, resume_mode=False, start_from=0, stop_on_multiprocessing_fail=True):
    """
    Generate images using streaming batch processing - BATCH ONLY VERSION (NO COMBINATION)
    """
    num_data = config['num_data']
    
    # Get memory management settings from config with safe defaults
    memory_config = config.get('memory_management', {})
    batch_size = memory_config.get('batch_size', config.get('batch_size', 50))
    aggressive_gc = memory_config.get('aggressive_gc', True)
    monitor_memory = memory_config.get('monitor_memory', False)
    
    save_folder = f"/scratch/regier_root/regier0/taodingr/descwl-shear-sims/generated_output/{config['setting']}"
    
    print(f"🧠 BATCH-ONLY GENERATION (HOME DIRECTORY)")
    print(f"   Save location: {save_folder}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total images: {num_data}")
    print(f"   Total batches: {(num_data + batch_size - 1) // batch_size}")
    print(f"   Aggressive GC: {aggressive_gc}")
    print(f"   ⚠️  COMBINATION DISABLED - Use separate script to combine")
    
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
        print(f"💾 Initial memory status: {get_memory_status()}")
    
    # Basic setup
    rng = np.random.RandomState(config['seed'])
    input_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
    print(f"The input image dimension is {input_dim}")
    
    # Pre-generate all shear values
    if config['shear_setting'] == "const":
        const = 0.05
        g1 = np.full(num_data, const, dtype=np.float32)
        g2 = np.full(num_data, const, dtype=np.float32)
    elif config['shear_setting'] == "vary":
        rng_shear = np.random.RandomState(config['seed'] + 1000)
        g1 = np.clip(rng_shear.normal(0.0, 0.015, num_data), -0.05, 0.05).astype(np.float32)
        g2 = np.clip(rng_shear.normal(0.0, 0.015, num_data), -0.05, 0.05).astype(np.float32)
    
    # Create star catalog
    star_catalog = None
    if config.get('star_catalog') is not None:
        star_catalog = make_star_catalog(
            rng=rng,
            coadd_dim=config['coadd_dim'],
            buff=config['buff'],
            pixel_scale=config['pixel_scale'],
            star_config=config['star_config']
        )
        if config.get('star_filter', False):
            star_catalog = filter_bright_stars_production(
                star_catalog, 
                mag_threshold=config['star_filter_mag'], 
                band=config['star_filter_band'], 
                verbose=True)
    
    # Create layout
    layout = Layout(
        layout_name=config['layout_name'],
        coadd_dim=config['coadd_dim'],
        pixel_scale=config['pixel_scale'],
        buff=config['buff']
    )
    
    shifts = layout.get_shifts(rng, density=config['density'])
    
    # Process in batches - BATCH GENERATION ONLY
    successful_batches = 0
    total_batches = (num_data + batch_size - 1) // batch_size
    
    for batch_start in range(0, num_data, batch_size):
        batch_end = min(batch_start + batch_size, num_data)
        batch_num = batch_start // batch_size + 1
        
        print(f"\n=== BATCH {batch_num}/{total_batches} ===")
        print(f"Processing images {batch_start} to {batch_end-1}")
        
        # Memory check before starting new batch
        if monitor_memory:
            print(f"💾 Memory before batch: {get_memory_status()}")
        
        # Start new batch (clears previous batch from memory)
        batch_manager.start_new_batch(batch_start)
        
        # Generate images for current batch
        batch_indices = list(range(batch_start, batch_end))
        
        print(f"Generating {len(batch_indices)} images...")
        
        # ===== SEQUENTIAL PROCESSING (Most Reliable) =====
        if not use_multiprocessing and not use_threading:
            print("Using sequential processing")
            
            # Create PSF once for sequential processing
            se_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
            psf = create_psf_for_worker(config, se_dim, config['seed'])
            
            for iter_idx in tqdm(batch_indices, desc=f"Batch {batch_num}"):
                each_image, positions_tensor, M, magnitude = Generate_single_img_catalog(
                    1, rng, config['mag'], config['hlr'], psf, config['morph'], 
                    config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
                    config['sep'], g1[iter_idx], g2[iter_idx], config['bands'], 
                    config['noise_factor'], config['dither'], config['dither_size'], 
                    config['rotate'], config['cosmic_rays'], config['bad_columns'], 
                    config['star_bleeds'], star_catalog, shifts, 
                    config['catalog_type'], config['select_observable'], 
                    config['select_lower_limit'], config['select_upper_limit'], config['draw_bright']
                )
                
                # Crop image
                H, W = each_image.shape[1], each_image.shape[2]
                crop_size = 2048
                start_h = (H - crop_size) // 2
                start_w = (W - crop_size) // 2
                single_image = each_image[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
                
                batch_manager.add_image_to_batch(iter_idx, single_image, positions_tensor, M, g1[iter_idx], g2[iter_idx])
                
                if aggressive_gc and iter_idx % 3 == 0:
                    import gc
                    gc.collect()
        
        # ===== MULTIPROCESSING (If Enabled) =====
        elif use_multiprocessing:
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
                        rng_state, psf_config, layout, shifts, star_catalog
                    ))
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    future_to_idx = {executor.submit(process_single_image_for_accumulation_fixed, args): args[0] for args in args_list}
                    
                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_idx),
                        total=len(batch_indices),
                        desc=f"Batch {batch_num} (MP)",
                        unit="img"
                    ):
                        iter_idx, single_image, positions_tensor, M, magnitude = future.result()
                        batch_manager.add_image_to_batch(iter_idx, single_image, positions_tensor, M, g1[iter_idx], g2[iter_idx])
                        
                        if aggressive_gc and iter_idx % 5 == 0:
                            import gc
                            gc.collect()
                
                print(f"✅ Multiprocessing completed for batch {batch_num}")
            
            except Exception as e:
                print(f"❌ Multiprocessing failed: {e}")
                if stop_on_multiprocessing_fail:
                    print("Stopping due to multiprocessing failure")
                    break
        
        # ===== THREADING (If Enabled) =====
        elif use_threading:
            if n_workers is None:
                n_workers = min(4, len(batch_indices))
            
            print(f"Using threading with {n_workers} workers")
            
            def process_batch_image(iter_idx):
                thread_rng = np.random.RandomState(config['seed'] + iter_idx * 1000)
                se_dim = get_se_dim(coadd_dim=config['coadd_dim'], rotate=config['rotate'])
                thread_psf = create_psf_for_worker(config, se_dim, config['seed'] + iter_idx)
                
                each_image, positions_tensor, M, magnitude = Generate_single_img_catalog(
                    1, thread_rng, config['mag'], config['hlr'], thread_psf, config['morph'], 
                    config['pixel_scale'], layout, config['coadd_dim'], config['buff'], 
                    config['sep'], g1[iter_idx], g2[iter_idx], config['bands'], 
                    config['noise_factor'], config['dither'], config['dither_size'], 
                    config['rotate'], config['cosmic_rays'], config['bad_columns'], 
                    config['star_bleeds'], star_catalog, shifts,
                    config['catalog_type'], config['select_observable'], config['select_lower_limit'], 
                    config['select_upper_limit'], config['draw_bright']
                )
                
                # Crop image
                H, W = each_image.shape[1], each_image.shape[2]
                crop_size = 2048
                start_h = (H - crop_size) // 2
                start_w = (W - crop_size) // 2
                single_image = each_image[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
                
                return iter_idx, single_image, positions_tensor, M, magnitude
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {executor.submit(process_batch_image, idx): idx for idx in batch_indices}
                
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_idx),
                    total=len(batch_indices),
                    desc=f"Batch {batch_num} (threading)",
                    unit="img"
                ):
                    iter_idx, single_image, positions_tensor, M, magnitude = future.result()
                    batch_manager.add_image_to_batch(iter_idx, single_image, positions_tensor, M, g1[iter_idx], g2[iter_idx])
                    
                    if aggressive_gc and iter_idx % 5 == 0:
                        import gc
                        gc.collect()
        
        # Save current batch to disk and clear from memory
        batch_save_success = batch_manager.save_current_batch_to_disk_improved()
        
        if batch_save_success:
            successful_batches += 1
            print(f"✅ Batch {batch_num} saved successfully")
        else:
            print(f"❌ Batch {batch_num} save failed")
            break
        
        # Force garbage collection after each batch
        if aggressive_gc:
            import gc
            gc.collect()
        
        if monitor_memory:
            print(f"💾 Memory after batch: {get_memory_status()}")
    
    # Final summary - NO COMBINATION
    print(f"\n=== BATCH GENERATION COMPLETED ===")
    print(f"✅ Successfully generated: {successful_batches}/{total_batches} batches")
    print(f"📁 Batch files location: {save_folder}")
    print(f"📋 Batch files pattern: batch_X_images.pt, batch_X_catalog.pt")
    
    
    return {
        'successful_batches': successful_batches,
        'total_batches': total_batches,
        'save_folder': save_folder,
        'setting': config['setting']
    }

def plot_magnitude_distribution(magnitudes, num_selected, config):
    if num_selected > config['num_data']:
        print(f"selected number of galaxies is greater than generated number of galaxies, setting num_selected to {config['num_data']}")
        num_selected = config['num_data']

    mag = magnitudes[:num_selected]
    mag_combined = np.concatenate(mag)
    plt.hist(mag_combined, bins=100)
    plt.xlabel("i-band ab magnitude")
    plt.ylabel("Count")
    plt.savefig(f"/scratch/regier_root/regier0/taodingr/descwl-shear-sims/notebooks/magnitude_distribution.png")

def load_tiled_tensor_data(setting, data_folder="/scratch/regier_root/regier0/taodingr/descwl-shear-sims/generated_output"):
    """
    NEW FUNCTION: Load the tiled tensor data
    
    Returns:
        images: torch.Tensor of shape [num_images, channels, height, width]
        catalog: dict with tiled structure
    """
    save_folder = f"{data_folder}/{setting}"
    image_file = f"{save_folder}/images_{setting}.pt"
    catalog_file = f"{save_folder}/catalog_{setting}.pt"
    
    print(f"Loading tiled images from: {image_file}")
    print(f"Loading tiled catalog from: {catalog_file}")
    
    # Load tensors
    images = torch.load(image_file, weights_only=True, map_location='cpu')
    catalog = torch.load(catalog_file, weights_only=True, map_location='cpu')
    
    print(f"✅ Loaded successfully!")
    print(f"Images shape: {images.shape}")
    print(f"Catalog structure:")
    for key, tensor in catalog.items():
        print(f"  {key}: {tensor.shape}")
    
    # Verify tiled structure
    n_images = catalog['n_sources'].shape[0]
    n_tiles_per_side = catalog['n_sources'].shape[1]
    total_tiles = n_tiles_per_side ** 2
    
    print(f"Number of images: {n_images}")
    print(f"Tiles per side: {n_tiles_per_side}")
    print(f"Total tiles per image: {total_tiles}")
    
    # Show some statistics
    total_sources_per_image = catalog['n_sources'].sum(dim=(1, 2))
    print(f"Sources per image - mean: {total_sources_per_image.float().mean():.1f}, "
          f"std: {total_sources_per_image.float().std():.1f}")
    
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

    magnitudes = Generate_img_catalog_batched_streaming_fixed(
        config, 
        use_multiprocessing=use_multiprocessing, 
        use_threading=use_threading, 
        n_workers=n_workers,
        resume_mode=resume_mode, 
        start_from=start_from,
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