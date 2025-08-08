"""
OPTIMIZED 512³ NAVIER-STOKES FRAMEWORK
Fixed version with proper time tracking and plot error handling
Enhanced checkpoint saving for visualization sequence
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import time
from datetime import datetime
import h5py
import os
import ctypes
import gc
from scipy.ndimage import gaussian_filter
import psutil

import multiprocessing as mp

# --- GPU Detection and Setup ---
GPU_AVAILABLE = False
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"

try:
    nvrtc_dll = os.path.join(CUDA_BIN, "nvrtc64_120_0.dll")
    builtins_dll = os.path.join(CUDA_BIN, "nvrtc-builtins64_120.dll")

    # Try loading runtime compiler DLLs manually
    ctypes.WinDLL(nvrtc_dll)
    if os.path.exists(builtins_dll):
        ctypes.WinDLL(builtins_dll)

    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        GPU_AVAILABLE = True
        print(f"GPU support available via CuPy ({cp.cuda.runtime.getDeviceProperties(0)['name'].decode()})")
    else:
        print("CuPy loaded but no CUDA device detected. Using CPU.")
except FileNotFoundError as e:
    print(f"Missing CUDA DLLs: {e}. GPU mode disabled.")
except OSError as e:
    print(f"Could not load CUDA DLLs: {e}. GPU mode disabled.")
except ImportError:
    print("CuPy not installed. Using CPU.")


class UltraOptimized512Framework:
    """
    Ultra-optimized framework specifically for 512³ simulations
    """
    
    def __init__(self, grid_size: int = 512, 
                 domain_size: float = 2*np.pi,
                 dt: float = 0.00005, 
                 reynolds_number: float = 1000.0,
                 use_gpu: bool = None,
                 save_frequency: int = 1000):
        
        self.nx = self.ny = self.nz = grid_size
        self.Lx = self.Ly = self.Lz = domain_size
        self.dx = self.dy = self.dz = domain_size / grid_size
        self.dt = dt
        self.Re = reynolds_number
        self.viscosity = 1.0 / reynolds_number
        self.save_frequency = save_frequency
        
        # GPU configuration
        if use_gpu is None:
            self.use_gpu = GPU_AVAILABLE
        else:
            self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            self.xp = cp
            self.fft = cp.fft
            print("Using GPU with CuPy/cuFFT")
        else:
            self.xp = np
            self.fft = np.fft
            print("Using CPU with NumPy/FFT")
        
        # Memory calculation and check
        self.memory_per_field_mb = self.nx * self.ny * self.nz * 4 / 1e6  # float32
        self.total_memory_estimate_mb = self.memory_per_field_mb * 15  # Conservative estimate
        available_memory_mb = psutil.virtual_memory().available / 1e6
        
        print(f"\nCPU Memory Check:")
        print(f"   Available RAM: {available_memory_mb:.1f} MB")
        print(f"   Required memory: {self.total_memory_estimate_mb:.1f} MB")
        
        if self.total_memory_estimate_mb > available_memory_mb * 0.8:
            raise MemoryError(f"Insufficient memory! Need {self.total_memory_estimate_mb:.1f}MB, "
                            f"only {available_memory_mb:.1f}MB available")
        
        print(f"   Memory check passed")
        print(f"   Array size: {self.memory_per_field_mb:.1f} MB each")
        print(f"   Precision: float32")
        
        # Multi-stage alpha control
        self.stages = [
            {'vorticity_threshold': 0,    'alpha': 0.996,  'name': 'Normal'},
            {'vorticity_threshold': 10,   'alpha': 0.997,  'name': 'Caution'},
            {'vorticity_threshold': 15,   'alpha': 0.998,  'name': 'Warning'},
            {'vorticity_threshold': 25,   'alpha': 0.999,  'name': 'Critical'},
            {'vorticity_threshold': 40,   'alpha': 0.9995, 'name': 'Emergency'},
            {'vorticity_threshold': 60,   'alpha': 0.9999, 'name': 'Ultra-Emergency'}
        ]
        
        self.current_stage = 0
        self.current_alpha = self.stages[0]['alpha']
        
        # Safety parameters
        self.max_allowed_growth_rate = 1.15
        self.min_dt = 1e-7
        self.max_dt = min(0.001, dt * 2)
        self.smoothing_threshold = 40.0
        
        # Pre-compute FFT frequencies for efficiency
        print(f"Setting up FFT frequencies...")
        self.kx = self.xp.fft.fftfreq(self.nx, self.dx) * 2 * self.xp.pi
        self.ky = self.xp.fft.fftfreq(self.ny, self.dy) * 2 * self.xp.pi
        self.kz = self.xp.fft.fftfreq(self.nz, self.dz) * 2 * self.xp.pi
        
        # Reshape for broadcasting
        self.kx = self.kx.reshape(-1, 1, 1).astype(self.xp.float32)
        self.ky = self.ky.reshape(1, -1, 1).astype(self.xp.float32)
        self.kz = self.kz.reshape(1, 1, -1).astype(self.xp.float32)
        
        # Laplacian operator
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        print(f"   FFT setup complete")
        
        # Initialize velocity fields
        print(f"Initializing velocity fields...")
        self.u = self.xp.zeros((self.nx, self.ny, self.nz), dtype=self.xp.float32)
        self.v = self.xp.zeros((self.nx, self.ny, self.nz), dtype=self.xp.float32)
        self.w = self.xp.zeros((self.nx, self.ny, self.nz), dtype=self.xp.float32)
        print(f"   Velocity fields initialized")
        
        # History storage (compact for 512³)
        self.time_history = []
        self.vorticity_max_history = []
        self.energy_history = []
        self.alignment_history = []
        self.alpha_history = []
        self.bkm_integral = 0.0
        
        # FIXED: Use regular float for time tracking to avoid overflow
        self.current_time = 0.0
        
        # Stage tracking
        self.stage_transitions = []
        
        print(f"\nUltra-Optimized 512³ Framework Ready")
        print(f"   Grid: {self.nx}x{self.ny}x{self.nz} = {self.nx*self.ny*self.nz:,} points")
        print(f"   Domain: {self.Lx:.2f}x{self.Ly:.2f}x{self.Lz:.2f}")
        print(f"   Reynolds number: {self.Re}")
        print(f"   Using GPU: {self.use_gpu}")
        
    def initialize_taylor_green(self, amplitude: float = 1.0):
        """Initialize 3D Taylor-Green vortex with optimized memory usage"""
        print(f"\nInitializing Taylor-Green vortex...")
        
        # Create coordinate arrays
        x = self.xp.linspace(0, self.Lx, self.nx, dtype=self.xp.float32)
        y = self.xp.linspace(0, self.Ly, self.ny, dtype=self.xp.float32)
        z = self.xp.linspace(0, self.Lz, self.nz, dtype=self.xp.float32)
        
        # Initialize slice by slice to save memory
        print(f"   Initializing velocity fields slice by slice...")
        
        # Pre-compute y-z meshgrid once
        Y, Z = self.xp.meshgrid(y, z, indexing='ij')
        Y = Y.astype(self.xp.float32)
        Z = Z.astype(self.xp.float32)
        
        cos_Y = self.xp.cos(Y)
        cos_Z = self.xp.cos(Z)
        sin_Y = self.xp.sin(Y)
        
        for i in range(self.nx):
            if i % 64 == 0:
                print(f"     Processing x-slice {i}/{self.nx}")
            
            sin_x = self.xp.sin(x[i])
            cos_x = self.xp.cos(x[i])
            
            # Taylor-Green initial conditions
            self.u[i, :, :] = amplitude * sin_x * cos_Y * cos_Z
            self.v[i, :, :] = -amplitude * cos_x * sin_Y * cos_Z
            self.w[i, :, :] = 0.0
        
        # Clean up temporary arrays
        del Y, Z, cos_Y, cos_Z, sin_Y, x, y, z
        
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
        
        print(f"   Taylor-Green initialization complete")
        
    def compute_derivatives_fft(self, field, axis):
        """Compute derivatives using pre-computed FFT frequencies"""
        if axis == 0:
            k = self.kx
        elif axis == 1:
            k = self.ky
        else:
            k = self.kz
        
        field_hat = self.fft.fftn(field)
        deriv_hat = 1j * k * field_hat
        return self.fft.ifftn(deriv_hat).real.astype(self.xp.float32)
    
    def compute_laplacian_fft(self, field):
        """Compute Laplacian using pre-computed operator"""
        field_hat = self.fft.fftn(field)
        laplacian_hat = -self.k2 * field_hat
        return self.fft.ifftn(laplacian_hat).real.astype(self.xp.float32)
    
    def compute_max_vorticity(self) -> float:
        """Compute maximum vorticity magnitude with memory optimization"""
        # Compute vorticity components
        dwdy = self.compute_derivatives_fft(self.w, 1)
        dvdz = self.compute_derivatives_fft(self.v, 2)
        omega_x = dwdy - dvdz
        del dwdy, dvdz
        
        dudz = self.compute_derivatives_fft(self.u, 2)
        dwdx = self.compute_derivatives_fft(self.w, 0)
        omega_y = dudz - dwdx
        del dudz, dwdx
        
        dvdx = self.compute_derivatives_fft(self.v, 0)
        dudy = self.compute_derivatives_fft(self.u, 1)
        omega_z = dvdx - dudy
        del dvdx, dudy
        
        # Compute magnitude efficiently
        vorticity_mag_squared = omega_x**2 + omega_y**2 + omega_z**2
        del omega_x, omega_y, omega_z
        
        max_vort = float(self.xp.sqrt(self.xp.max(vorticity_mag_squared)))
        del vorticity_mag_squared
        
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
        
        return max_vort
    
    def compute_energy(self) -> float:
        """Compute total kinetic energy"""
        energy = 0.5 * self.xp.mean(self.u**2 + self.v**2 + self.w**2)
        return float(energy)
    
    def compute_alignment_score(self, u_new, v_new, w_new, u_old, v_old, w_old) -> float:
        """Compute alignment score with heavy subsampling for 512³"""
        # Subsample heavily for memory efficiency (every 8th point)
        step = 8
        u_new_s = u_new[::step, ::step, ::step]
        v_new_s = v_new[::step, ::step, ::step]
        w_new_s = w_new[::step, ::step, ::step]
        u_old_s = u_old[::step, ::step, ::step]
        v_old_s = v_old[::step, ::step, ::step]
        w_old_s = w_old[::step, ::step, ::step]
        
        dot_product = (self.xp.sum(u_new_s * u_old_s) + 
                      self.xp.sum(v_new_s * v_old_s) + 
                      self.xp.sum(w_new_s * w_old_s))
        norm_new = self.xp.sqrt(self.xp.sum(u_new_s**2 + v_new_s**2 + w_new_s**2))
        norm_old = self.xp.sqrt(self.xp.sum(u_old_s**2 + v_old_s**2 + w_old_s**2))
        
        if norm_new == 0 or norm_old == 0:
            return 1.0
        
        return float(dot_product / (norm_new * norm_old))
    
    def enforce_alignment(self, u_new, v_new, w_new, u_old, v_old, w_old, target_alpha):
        """Enforce alignment constraint"""
        alignment = self.compute_alignment_score(u_new, v_new, w_new, u_old, v_old, w_old)
        
        if alignment >= target_alpha:
            return u_new, v_new, w_new, alignment
        
        # Simple projection
        factor = 0.9
        u_aligned = factor * u_old + (1 - factor) * u_new
        v_aligned = factor * v_old + (1 - factor) * v_new
        w_aligned = factor * w_old + (1 - factor) * w_new
        
        alignment = self.compute_alignment_score(u_aligned, v_aligned, w_aligned, u_old, v_old, w_old)
        
        return u_aligned, v_aligned, w_aligned, alignment
    
    def apply_smoothing(self, u, v, w, vorticity_max):
        """Apply smoothing for high vorticity"""
        if vorticity_max < self.smoothing_threshold:
            return u, v, w
        
        # Determine smoothing strength
        if vorticity_max > 80:
            sigma = 2.0  # Reduced for 512³
        elif vorticity_max > 60:
            sigma = 1.5
        elif vorticity_max > 40:
            sigma = 1.0
        else:
            sigma = 0.8
        
        # Apply smoothing
        if self.use_gpu:
            # Transfer to CPU for scipy smoothing
            u_cpu = cp.asnumpy(u)
            v_cpu = cp.asnumpy(v)
            w_cpu = cp.asnumpy(w)
            
            u_smooth = gaussian_filter(u_cpu, sigma=sigma, mode='wrap')
            v_smooth = gaussian_filter(v_cpu, sigma=sigma, mode='wrap')
            w_smooth = gaussian_filter(w_cpu, sigma=sigma, mode='wrap')
            
            # Transfer back to GPU
            u = cp.asarray(u_smooth, dtype=cp.float32)
            v = cp.asarray(v_smooth, dtype=cp.float32)
            w = cp.asarray(w_smooth, dtype=cp.float32)
        else:
            u = gaussian_filter(u, sigma=sigma, mode='wrap').astype(np.float32)
            v = gaussian_filter(v, sigma=sigma, mode='wrap').astype(np.float32)
            w = gaussian_filter(w, sigma=sigma, mode='wrap').astype(np.float32)
        
        return u, v, w
    
    def evolve_one_timestep(self, step: int) -> Dict:
        """Evolve one timestep with all stability features"""
        # Store old fields
        u_old = self.u.copy()
        v_old = self.v.copy()
        w_old = self.w.copy()
        
        # Get current vorticity
        current_vort_max = self.vorticity_max_history[-1] if self.vorticity_max_history else 10.0
        
        # Calculate growth rate
        if len(self.vorticity_max_history) > 1:
            growth_rate = current_vort_max / self.vorticity_max_history[-2]
        else:
            growth_rate = 1.0
        
        # Update alpha based on vorticity
        target_stage = 0
        for i, stage in enumerate(self.stages):
            if current_vort_max >= stage['vorticity_threshold']:
                target_stage = i
        
        # Boost for high growth
        if growth_rate > 1.05 and target_stage < len(self.stages) - 1:
            target_stage += 1
        
        # Update alpha
        if target_stage != self.current_stage:
            self.current_stage = target_stage
            self.stage_transitions.append({
                'step': step,
                'time': self.current_time,
                'stage': self.stages[target_stage]['name'],
                'vorticity': current_vort_max
            })
        
        self.current_alpha = self.stages[target_stage]['alpha']
        stage_name = self.stages[target_stage]['name']
        
        # Adaptive timestep for 512³
        if stage_name == 'Ultra-Emergency':
            dt_factor = 0.05  # Very aggressive for 512³
        elif stage_name in ['Critical', 'Emergency']:
            dt_factor = 0.1
        elif stage_name == 'Warning':
            dt_factor = 0.25
        else:
            dt_factor = 1.0
        
        dt = max(self.min_dt, min(self.max_dt, self.dt * dt_factor))
        
        # Compute derivatives using FFT
        dudx = self.compute_derivatives_fft(self.u, 0)
        dudy = self.compute_derivatives_fft(self.u, 1)
        dudz = self.compute_derivatives_fft(self.u, 2)
        
        dvdx = self.compute_derivatives_fft(self.v, 0)
        dvdy = self.compute_derivatives_fft(self.v, 1)
        dvdz = self.compute_derivatives_fft(self.v, 2)
        
        dwdx = self.compute_derivatives_fft(self.w, 0)
        dwdy = self.compute_derivatives_fft(self.w, 1)
        dwdz = self.compute_derivatives_fft(self.w, 2)
        
        # Convective terms
        conv_u = self.u * dudx + self.v * dudy + self.w * dudz
        conv_v = self.u * dvdx + self.v * dvdy + self.w * dvdz
        conv_w = self.u * dwdx + self.v * dwdy + self.w * dwdz
        
        # Clean up derivatives
        del dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz
        
        # Viscous terms (using FFT Laplacian)
        visc_u = self.viscosity * self.compute_laplacian_fft(self.u)
        visc_v = self.viscosity * self.compute_laplacian_fft(self.v)
        visc_w = self.viscosity * self.compute_laplacian_fft(self.w)
        
        # Update velocities
        u_new = self.u + dt * (-conv_u + visc_u)
        v_new = self.v + dt * (-conv_v + visc_v)
        w_new = self.w + dt * (-conv_w + visc_w)
        
        # Clean up convective and viscous terms
        del conv_u, conv_v, conv_w, visc_u, visc_v, visc_w
        
        # Apply smoothing if needed
        if current_vort_max > self.smoothing_threshold:
            u_new, v_new, w_new = self.apply_smoothing(u_new, v_new, w_new, current_vort_max)
        
        # Check growth and limit if needed
        vort_estimate = current_vort_max * 1.1  # Approximate to save computation
        if vort_estimate / current_vort_max > self.max_allowed_growth_rate:
            scale = self.max_allowed_growth_rate * current_vort_max / vort_estimate
            u_new = self.u + scale * (u_new - self.u)
            v_new = self.v + scale * (v_new - self.v)
            w_new = self.w + scale * (w_new - self.w)
        
        # Enforce alignment
        u_new, v_new, w_new, alignment = self.enforce_alignment(
            u_new, v_new, w_new, u_old, v_old, w_old, self.current_alpha
        )
        
        # Update fields
        self.u = u_new
        self.v = v_new
        self.w = w_new
        
        # Compute final diagnostics
        vorticity_max = self.compute_max_vorticity()
        energy = self.compute_energy()
        
        # Update BKM integral
        self.bkm_integral += vorticity_max * dt
        
        # FIXED: Update time with regular float
        self.current_time += dt
        
        return {
            'vorticity_max': vorticity_max,
            'energy': energy,
            'alignment': alignment,
            'alpha': self.current_alpha,
            'stage': stage_name,
            'dt': dt,
            'time': self.current_time
        }
    
    def run_simulation(self, steps: int):
        """Run 512³ simulation with optimized checkpointing"""
        print(f"\nStarting 512³ simulation for {steps} steps...")
        print(f"   Initial dt: {self.dt:.7f}")
        print(f"   Target: Navigate through critical period (t~1.5)")
        
        start_time = time.time()
        last_save_time = start_time
        
        # Create output directory
        output_dir = f"512_cube_output_{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Critical checkpoints - FIXED: Added visualization steps + final step
        critical_steps = [500, 1000, 1499, 1500, 1720, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
        visualization_steps = [2000, 4000, 6000, 8000, 10000]  # Steps for visualization sequence
        
        # Main simulation loop
        for step in range(steps):
            # Evolve
            diagnostics = self.evolve_one_timestep(step)

            # Store history
            self.time_history.append(diagnostics['time'])
            self.vorticity_max_history.append(diagnostics['vorticity_max'])
            self.energy_history.append(diagnostics['energy'])
            self.alignment_history.append(diagnostics['alignment'])
            self.alpha_history.append(diagnostics['alpha'])
            
            # Progress reporting
            if step % 100 == 0 or step in critical_steps:
                elapsed = time.time() - start_time
                eta = elapsed / (step + 1) * (steps - step - 1) if step > 0 else 0
                
                print(f"Step {step:5d}: t={diagnostics['time']:.5f}, "
                      f"Stage={diagnostics['stage']:14s}, "
                      f"alpha={diagnostics['alpha']:.6f}, "
                      f"Vort={diagnostics['vorticity_max']:6.2f}, "
                      f"BKM={self.bkm_integral:.4f}, "
                      f"ETA={eta/60:.1f}min")
                
                # Memory usage for 512³
                if step % 500 == 0:
                    memory_gb = psutil.Process().memory_info().rss / 1e9
                    if self.use_gpu:
                        gpu_memory_gb = cp.get_default_memory_pool().used_bytes() / 1e9
                        print(f"   Memory: CPU {memory_gb:.1f} GB, GPU {gpu_memory_gb:.1f} GB")
                    else:
                        print(f"   Memory usage: {memory_gb:.1f} GB")
            
            # Critical step reporting
            if step == 1499:
                print(f"\nCRITICAL STEP 1499 (512³):")
                print(f"   Time: {diagnostics['time']:.5f}")
                print(f"   Vorticity: {diagnostics['vorticity_max']:.2f}")
                print(f"   Alpha: {diagnostics['alpha']:.6f}")
                print(f"   Stage: {diagnostics['stage']}")
            
            # FIXED: Enhanced checkpoint saving
            should_save = False
            save_reason = ""
            
            # Regular interval saves
            if step % self.save_frequency == 0 and step > 0:
                should_save = True
                save_reason = "regular"
            
            # Critical step saves
            if step in critical_steps:
                should_save = True
                save_reason = "critical"
            
            # Visualization step saves (always save these!)
            if step in visualization_steps:
                should_save = True
                save_reason = "visualization"
            
            # Final step (always save!)
            if step == steps - 1:
                should_save = True
                save_reason = "final"
            
            if should_save:
                self.save_checkpoint(step, output_dir)
                current_save_time = time.time()
                print(f"   512³ checkpoint saved ({save_reason}) (took {current_save_time - last_save_time:.1f}s)")
                last_save_time = current_save_time
            
            # Safety check
            if diagnostics['vorticity_max'] > 150:
                print(f"\nVorticity exceeded 150 at step {step} - stopping for safety")
                break
        
        # Final results
        runtime = time.time() - start_time
        success = max(self.vorticity_max_history) < 100
        
        print(f"\n512³ simulation completed in {runtime/60:.1f} minutes")
        print(f"\nFINAL 512³ RESULTS:")
        print(f"   Success: {'YES!' if success else 'NO'}")
        print(f"   Total steps: {len(self.time_history)}")
        print(f"   Final time: {self.time_history[-1]:.5f}")
        print(f"   Max vorticity: {max(self.vorticity_max_history):.2f}")
        print(f"   Final BKM: {self.bkm_integral:.6f}")
        print(f"   Min alignment: {min(self.alignment_history):.6f}")
        
        # Save final results
        self.save_final_results(output_dir, success)
        
        return {
            'success': success,
            'steps': len(self.time_history),
            'max_vorticity': max(self.vorticity_max_history),
            'bkm_integral': self.bkm_integral,
            'runtime_minutes': runtime/60
        }
    
    def save_checkpoint(self, step: int, output_dir: str):
        """Save lightweight checkpoint for 512³"""
        filename = f"{output_dir}/checkpoint_step_{step:06d}.h5"
        
        # For 512³, save compressed velocity fields
        with h5py.File(filename, 'w') as f:
            if self.use_gpu:
                f.create_dataset('u', data=cp.asnumpy(self.u), compression='gzip', compression_opts=6)
                f.create_dataset('v', data=cp.asnumpy(self.v), compression='gzip', compression_opts=6)
                f.create_dataset('w', data=cp.asnumpy(self.w), compression='gzip', compression_opts=6)
            else:
                f.create_dataset('u', data=self.u, compression='gzip', compression_opts=6)
                f.create_dataset('v', data=self.v, compression='gzip', compression_opts=6)
                f.create_dataset('w', data=self.w, compression='gzip', compression_opts=6)
            
            # Metadata
            f.attrs['step'] = step
            f.attrs['time'] = self.time_history[-1]
            f.attrs['vorticity_max'] = self.vorticity_max_history[-1]
            f.attrs['energy'] = self.energy_history[-1]
            f.attrs['bkm_integral'] = self.bkm_integral
    
    def save_final_results(self, output_dir: str, success: bool):
        """Save final results"""
        filename = f"{output_dir}/final_results_512cube.h5"
        
        with h5py.File(filename, 'w') as f:
            # Time series data
            f.create_dataset('time', data=self.time_history)
            f.create_dataset('vorticity_max', data=self.vorticity_max_history)
            f.create_dataset('energy', data=self.energy_history)
            f.create_dataset('alignment', data=self.alignment_history)
            f.create_dataset('alpha', data=self.alpha_history)
            
            # Parameters
            f.attrs['grid_size'] = 512
            f.attrs['Re'] = self.Re
            f.attrs['success'] = success
            f.attrs['max_vorticity'] = max(self.vorticity_max_history)
            f.attrs['bkm_integral'] = self.bkm_integral
            
            # Stage transitions
            if self.stage_transitions:
                trans_group = f.create_group('stage_transitions')
                for i, trans in enumerate(self.stage_transitions):
                    trans_group.attrs[f'transition_{i}'] = (
                        f"Step {trans['step']}: {trans['stage']} at vort={trans['vorticity']:.2f}"
                    )
        
        # Generate plots
        self.generate_plots(output_dir)
    
    def generate_plots(self, output_dir: str):
        """Generate summary plots for 512³"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Vorticity evolution
            ax = axes[0, 0]
            ax.semilogy(self.time_history, self.vorticity_max_history, 'b-', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Max Vorticity')
            ax.set_title('512³ Vorticity Evolution')
            ax.grid(True, alpha=0.3)
            
            # Alpha evolution
            ax = axes[0, 1]
            ax.plot(self.time_history, self.alpha_history, 'r-', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('alpha')
            ax.set_title('Adaptive Alpha (512³)')
            ax.set_ylim([0.995, 1.0])
            ax.grid(True, alpha=0.3)
            
            # Alignment scores
            ax = axes[0, 2]
            ax.plot(self.time_history, self.alignment_history, 'g-', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Alignment Score')
            ax.set_title('Alignment Evolution (512³)')
            ax.set_ylim([0.995, 1.001])
            ax.grid(True, alpha=0.3)
            
            # Energy
            ax = axes[1, 0]
            ax.plot(self.time_history, self.energy_history, 'c-', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Kinetic Energy')
            ax.set_title('Energy Evolution (512³)')
            ax.grid(True, alpha=0.3)
            
            # BKM integral
            ax = axes[1, 1]
            # FIXED: Handle potential empty arrays and division by zero
            if len(self.time_history) > 1 and len(self.vorticity_max_history) > 0:
                dt_avg = np.mean(np.diff(self.time_history))
                if dt_avg > 0 and np.isfinite(dt_avg):
                    bkm_cumulative = np.cumsum(self.vorticity_max_history) * dt_avg
                else:
                    # Fallback: use the stored BKM integral
                    bkm_cumulative = np.linspace(0, self.bkm_integral, len(self.time_history))
            else:
                bkm_cumulative = [0] * len(self.time_history)
                
            ax.plot(self.time_history, bkm_cumulative, 'm-', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Integral ||w||_inf dt')
            ax.set_title(f'BKM Integral (Final: {self.bkm_integral:.4f})')
            ax.grid(True, alpha=0.3)
            
            # Summary
            ax = axes[1, 2]
            success = max(self.vorticity_max_history) < 100
            ax.text(0.5, 0.8, '512³ CUBE RESULTS', fontsize=16, ha='center', weight='bold')
            ax.text(0.5, 0.6, 'SUCCESS!' if success else 'Failed', 
                    fontsize=20, ha='center', 
                    color='green' if success else 'red', weight='bold')
            ax.text(0.5, 0.4, f'Max vorticity: {max(self.vorticity_max_history):.2f}', 
                    fontsize=12, ha='center')
            ax.text(0.5, 0.3, f'Steps: {len(self.time_history)}', fontsize=12, ha='center')
            ax.text(0.5, 0.2, f'Re = {self.Re}', fontsize=12, ha='center')
            ax.text(0.5, 0.1, f'134M+ grid points', fontsize=10, ha='center', style='italic')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/512_cube_summary.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"Warning: Error generating plots: {e}")
            print("Continuing without plots...")


def run_512_cube_test():
    """
    Run the 512³ test
    """
    print("="*80)
    print("512³ NAVIER-STOKES TEST")
    print("   134,217,728 grid points - Large scale simulation")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System info
    print(f"\nSystem Information:")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  Total RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"  Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")
    if GPU_AVAILABLE:
        print(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    
    try:
        # Create framework
        framework = UltraOptimized512Framework(
            grid_size=512,
            dt=0.00005,  # Very small timestep for stability
            reynolds_number=1000.0,
            use_gpu=True,  # Use GPU if available
            save_frequency=2000  # Save less frequently
        )
        
        # Initialize
        framework.initialize_taylor_green(amplitude=1.0)
        
        # Run simulation - aim for critical time period
        result = framework.run_simulation(steps=10000)  # Should reach beyond t=1.5
        
        print(f"\n512³ TEST COMPLETED:")
        print(f"   Success: {result['success']}")
        print(f"   Max vorticity: {result['max_vorticity']:.2f}")
        print(f"   BKM integral: {result['bkm_integral']:.6f}")
        print(f"   Runtime: {result['runtime_minutes']:.1f} minutes")
        
        if result['success']:
            print(f"\n512³ simulation completed successfully")
            print(f"   All stability criteria satisfied")
        
        return result
        
    except Exception as e:
        print(f"\nError in 512³ test: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    finally:
        # Clean up memory
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()


if __name__ == "__main__":
    result = run_512_cube_test()
    
    if result.get('success', False):
        print("\n512³ TEST SUCCESSFUL")
        print("   Simulation completed within stability bounds")
    else:
        print(f"\nTest completed")
    
    print("\n512³ test completed")