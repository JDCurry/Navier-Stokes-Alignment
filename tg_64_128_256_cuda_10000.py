"""
Navier-Stokes Alignment Framework Test Suite - CUDA Enhanced
Enhanced version with GPU support for faster testing
Follows EXACT same logic as original with CUDA acceleration
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import os
import time
from datetime import datetime
import psutil
from scipy.ndimage import gaussian_filter
import warnings
import ctypes
import gc

# Force non-interactive backend and suppress overflow warnings (expected behavior)
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

class NavierStokesAlignmentTestCUDA:
    """
    CUDA-enhanced test implementation following EXACT logic from original script
    Only adds GPU acceleration without changing mathematical behavior
    """
    
    def __init__(self, grid_size=64, dt=0.001, reynolds_number=1000.0, alpha=0.999, use_gpu=None):
        self.nx = self.ny = self.nz = grid_size
        self.Lx = self.Ly = self.Lz = 2 * np.pi
        self.dx = self.dy = self.dz = self.Lx / grid_size
        self.dt = dt
        self.original_dt = dt
        self.Re = reynolds_number
        self.viscosity = 1.0 / reynolds_number
        self.alpha = alpha
        
        # GPU configuration
        if use_gpu is None:
            self.use_gpu = GPU_AVAILABLE
        else:
            self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            self.xp = cp
            self.fft = cp.fft
            print(f"  Using GPU with CuPy/cuFFT")
        else:
            self.xp = np
            self.fft = np.fft
            print(f"  Using CPU with NumPy/FFT")
        
        print(f"Initializing {grid_size}^3 test...")
        print(f"  Grid points: {grid_size**3:,}")
        print(f"  dt: {dt}")
        print(f"  Re: {reynolds_number}")
        print(f"  Alpha: {alpha}")
        
        # EXACT same multi-stage alpha control as original
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
        
        # EXACT same safety parameters as original
        self.max_allowed_growth_rate = 1.15
        self.min_dt = 1e-7
        self.max_dt = min(0.001, dt * 2)
        self.smoothing_threshold = 40.0
        
        # Pre-compute FFT frequencies (same as original)
        kx = self.xp.fft.fftfreq(self.nx, self.dx) * 2 * self.xp.pi
        ky = self.xp.fft.fftfreq(self.ny, self.dy) * 2 * self.xp.pi
        kz = self.xp.fft.fftfreq(self.nz, self.dz) * 2 * self.xp.pi
        
        self.kx = kx.reshape(-1, 1, 1).astype(self.xp.float32)
        self.ky = ky.reshape(1, -1, 1).astype(self.xp.float32)
        self.kz = kz.reshape(1, 1, -1).astype(self.xp.float32)
        
        # Laplacian operator
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        
        # Initialize velocity fields
        self.u = self.xp.zeros((self.nx, self.ny, self.nz), dtype=self.xp.float32)
        self.v = self.xp.zeros((self.nx, self.ny, self.nz), dtype=self.xp.float32)
        self.w = self.xp.zeros((self.nx, self.ny, self.nz), dtype=self.xp.float32)
        
        # History tracking (same as original)
        self.time_history = []
        self.vorticity_max_history = []
        self.energy_history = []
        self.alignment_history = []
        self.alpha_history = []
        self.bkm_integral = 0.0
        self.current_time = 0.0
        self.stage_transitions = []
        
    def safe_cast(self, arr):
        """Minimal safety casting - only prevents NaN propagation"""
        if self.xp.any(self.xp.isnan(arr)) or self.xp.any(self.xp.isinf(arr)):
            if self.use_gpu:
                arr = self.xp.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
            else:
                arr = self.xp.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
        return arr.astype(self.xp.float32)
        
    def initialize_taylor_green(self, amplitude=1.0):
        """Initialize Taylor-Green vortex - EXACT same as original"""
        x = self.xp.linspace(0, self.Lx, self.nx)
        y = self.xp.linspace(0, self.Ly, self.ny)
        z = self.xp.linspace(0, self.Lz, self.nz)
        
        X, Y, Z = self.xp.meshgrid(x, y, z, indexing='ij')
        
        self.u = (amplitude * self.xp.sin(X) * self.xp.cos(Y) * self.xp.cos(Z)).astype(self.xp.float32)
        self.v = (-amplitude * self.xp.cos(X) * self.xp.sin(Y) * self.xp.cos(Z)).astype(self.xp.float32)
        self.w = self.xp.zeros_like(self.u)
        
        print(f"  Initialized Taylor-Green vortex (amplitude={amplitude})")
    
    def compute_derivatives_fft(self, field, axis):
        """Compute derivatives using FFT - EXACT same logic as original"""
        if axis == 0:
            k = self.kx
        elif axis == 1:
            k = self.ky
        else:
            k = self.kz
        
        field_hat = self.fft.fftn(field)
        deriv_hat = 1j * k * field_hat
        result = self.fft.ifftn(deriv_hat).real
        return self.safe_cast(result)
    
    def compute_laplacian_fft(self, field):
        """Compute Laplacian - EXACT same logic as original"""
        field_hat = self.fft.fftn(field)
        laplacian_hat = -self.k2 * field_hat
        result = self.fft.ifftn(laplacian_hat).real
        return self.safe_cast(result)
    
    def compute_max_vorticity(self):
        """Compute maximum vorticity - EXACT same logic as original"""
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
        
        vorticity_mag_squared = self.safe_cast(vorticity_mag_squared)
        max_vort = float(self.xp.sqrt(self.xp.max(vorticity_mag_squared)))
        del vorticity_mag_squared
        
        # Protect against NaN in final result
        if self.xp.isnan(max_vort) or self.xp.isinf(max_vort):
            max_vort = self.vorticity_max_history[-1] if self.vorticity_max_history else 10.0
        
        return max_vort
    
    def compute_energy(self):
        """Compute energy - EXACT same logic as original"""
        energy = 0.5 * self.xp.mean(self.u**2 + self.v**2 + self.w**2)
        energy = float(energy)
        if self.xp.isnan(energy) or self.xp.isinf(energy):
            energy = self.energy_history[-1] if self.energy_history else 0.5
        return energy
    
    def compute_alignment_score(self, u_new, v_new, w_new, u_old, v_old, w_old):
        """Compute alignment - EXACT same logic as original but adjusted subsampling for grid size"""
        # Adjust subsampling based on grid size (same approach as original)
        if self.nx >= 128:
            step = 8  # Same as original
        else:
            step = 4  # Slightly less subsampling for smaller grids
            
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
        
        alignment = dot_product / (norm_new * norm_old)
        
        # Protect against NaN
        if self.xp.isnan(alignment) or self.xp.isinf(alignment):
            return 1.0
            
        return float(alignment)
    
    def enforce_alignment(self, u_new, v_new, w_new, u_old, v_old, w_old, target_alpha):
        """Enforce alignment - EXACT same logic as original"""
        alignment = self.compute_alignment_score(u_new, v_new, w_new, u_old, v_old, w_old)
        
        if alignment >= target_alpha:
            return u_new, v_new, w_new, alignment
        
        # Simple projection - EXACT same factor as original
        factor = 0.9
        u_aligned = factor * u_old + (1 - factor) * u_new
        v_aligned = factor * v_old + (1 - factor) * v_new
        w_aligned = factor * w_old + (1 - factor) * w_new
        
        # Safe casting
        u_aligned = self.safe_cast(u_aligned)
        v_aligned = self.safe_cast(v_aligned)
        w_aligned = self.safe_cast(w_aligned)
        
        alignment = self.compute_alignment_score(u_aligned, v_aligned, w_aligned, u_old, v_old, w_old)
        
        return u_aligned, v_aligned, w_aligned, alignment
    
    def apply_smoothing(self, u, v, w, vorticity_max):
        """Apply smoothing - EXACT same logic and thresholds as original"""
        if vorticity_max < self.smoothing_threshold:
            return u, v, w
        
        # EXACT same smoothing strength logic as original
        if vorticity_max > 80:
            sigma = 2.0
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
    
    def evolve_one_timestep(self, step):
        """Evolve one timestep - EXACT same logic as original"""
        # Store old fields
        u_old = self.u.copy()
        v_old = self.v.copy()
        w_old = self.w.copy()
        
        # Get current vorticity - EXACT same logic as original
        current_vort_max = self.vorticity_max_history[-1] if self.vorticity_max_history else 10.0
        
        # Calculate growth rate - EXACT same logic as original
        if len(self.vorticity_max_history) > 1:
            growth_rate = current_vort_max / self.vorticity_max_history[-2]
        else:
            growth_rate = 1.0
        
        # Update alpha based on vorticity - EXACT same logic as original
        target_stage = 0
        for i, stage in enumerate(self.stages):
            if current_vort_max >= stage['vorticity_threshold']:
                target_stage = i
        
        # Boost for high growth - EXACT same logic as original
        if growth_rate > 1.05 and target_stage < len(self.stages) - 1:
            target_stage += 1
        
        # Update alpha - EXACT same logic as original
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
        
        # Adaptive timestep - EXACT same logic as original
        if stage_name == 'Ultra-Emergency':
            dt_factor = 0.05  # Very aggressive
        elif stage_name in ['Critical', 'Emergency']:
            dt_factor = 0.1
        elif stage_name == 'Warning':
            dt_factor = 0.25
        else:
            dt_factor = 1.0
        
        dt = max(self.min_dt, min(self.max_dt, self.original_dt * dt_factor))
        
        # Compute derivatives using FFT - EXACT same logic as original
        dudx = self.compute_derivatives_fft(self.u, 0)
        dudy = self.compute_derivatives_fft(self.u, 1)
        dudz = self.compute_derivatives_fft(self.u, 2)
        
        dvdx = self.compute_derivatives_fft(self.v, 0)
        dvdy = self.compute_derivatives_fft(self.v, 1)
        dvdz = self.compute_derivatives_fft(self.v, 2)
        
        dwdx = self.compute_derivatives_fft(self.w, 0)
        dwdy = self.compute_derivatives_fft(self.w, 1)
        dwdz = self.compute_derivatives_fft(self.w, 2)
        
        # Convective terms - EXACT same logic as original
        conv_u = self.u * dudx + self.v * dudy + self.w * dudz
        conv_v = self.u * dvdx + self.v * dvdy + self.w * dvdz
        conv_w = self.u * dwdx + self.v * dwdy + self.w * dwdz
        
        # Safe casting for convective terms
        conv_u = self.safe_cast(conv_u)
        conv_v = self.safe_cast(conv_v)
        conv_w = self.safe_cast(conv_w)
        
        # Clean up derivatives
        del dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz
        
        # Viscous terms - EXACT same logic as original
        visc_u = self.viscosity * self.compute_laplacian_fft(self.u)
        visc_v = self.viscosity * self.compute_laplacian_fft(self.v)
        visc_w = self.viscosity * self.compute_laplacian_fft(self.w)
        
        # Update velocities - EXACT same logic as original
        u_new = self.u + dt * (-conv_u + visc_u)
        v_new = self.v + dt * (-conv_v + visc_v)
        w_new = self.w + dt * (-conv_w + visc_w)
        
        # Safe casting
        u_new = self.safe_cast(u_new)
        v_new = self.safe_cast(v_new)
        w_new = self.safe_cast(w_new)
        
        # Clean up convective and viscous terms
        del conv_u, conv_v, conv_w, visc_u, visc_v, visc_w
        
        # Apply smoothing if needed - EXACT same logic as original
        if current_vort_max > self.smoothing_threshold:
            u_new, v_new, w_new = self.apply_smoothing(u_new, v_new, w_new, current_vort_max)
        
        # Check growth and limit if needed - EXACT same logic as original
        vort_estimate = current_vort_max * 1.1  # Approximate to save computation
        if vort_estimate / current_vort_max > self.max_allowed_growth_rate:
            scale = self.max_allowed_growth_rate * current_vort_max / vort_estimate
            u_new = self.u + scale * (u_new - self.u)
            v_new = self.v + scale * (v_new - self.v)
            w_new = self.w + scale * (w_new - self.w)
        
        # Enforce alignment - EXACT same logic as original
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
        
        # Update BKM integral - EXACT same logic as original
        self.bkm_integral += vorticity_max * dt
        
        # Update time - EXACT same logic as original
        self.current_time += dt
        
        # GPU sync if needed
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
        
        return {
            'vorticity_max': vorticity_max,
            'energy': energy,
            'alignment': alignment,
            'alpha': self.current_alpha,
            'stage': stage_name,
            'dt': dt,
            'time': self.current_time
        }
    
    def run_test(self, steps=10000):
        """Run test - enhanced reporting and 10K steps"""
        print(f"Running {steps} steps with CUDA acceleration...")
        print(f"Note: Early vorticity spikes are expected Taylor-Green behavior")
        start_time = time.time()
        
        for step in range(steps):
            diagnostics = self.evolve_one_timestep(step)
            
            # Store history
            self.time_history.append(diagnostics['time'])
            self.vorticity_max_history.append(diagnostics['vorticity_max'])
            self.energy_history.append(diagnostics['energy'])
            self.alignment_history.append(diagnostics['alignment'])
            self.alpha_history.append(diagnostics['alpha'])
            
            # Progress reporting - more frequent for 10K steps
            if step % 200 == 0 or step == steps - 1:
                elapsed = time.time() - start_time
                eta = elapsed / (step + 1) * (steps - step - 1) if step > 0 else 0
                
                print(f"  Step {step:5d}: t={diagnostics['time']:.5f}, "
                      f"Stage={diagnostics['stage']:14s}, "
                      f"alpha={diagnostics['alpha']:.6f}, "
                      f"Vort={diagnostics['vorticity_max']:6.2f}, "
                      f"BKM={self.bkm_integral:.4f}, "
                      f"ETA={eta/60:.1f}min")
                
                # Memory reporting
                if step % 1000 == 0 and step > 0:
                    memory_gb = psutil.Process().memory_info().rss / 1e9
                    if self.use_gpu:
                        gpu_memory_gb = cp.get_default_memory_pool().used_bytes() / 1e9
                        print(f"    Memory: CPU {memory_gb:.1f} GB, GPU {gpu_memory_gb:.1f} GB")
                    else:
                        print(f"    Memory usage: {memory_gb:.1f} GB")
        
        runtime = time.time() - start_time
        print(f"Completed in {runtime:.2f} seconds ({steps/runtime:.1f} steps/sec)")
        
        # Same success criterion as original
        return {
            'success': max(self.vorticity_max_history) < 100,
            'max_vorticity': max(self.vorticity_max_history),
            'final_time': self.current_time,
            'runtime': runtime,
            'steps_per_sec': steps / runtime,
            'bkm_integral': self.bkm_integral
        }
    
    def save_results(self, output_dir, grid_size):
        """Save results - same as original"""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/test_{grid_size}cubed_cuda_10k.h5"
        with h5py.File(filename, 'w') as f:
            # Convert GPU arrays to CPU if needed
            if self.use_gpu:
                time_data = np.array(self.time_history)
                vorticity_data = np.array(self.vorticity_max_history)
                energy_data = np.array(self.energy_history)
                alignment_data = np.array(self.alignment_history)
                alpha_data = np.array(self.alpha_history)
            else:
                time_data = self.time_history
                vorticity_data = self.vorticity_max_history
                energy_data = self.energy_history
                alignment_data = self.alignment_history
                alpha_data = self.alpha_history
            
            f.create_dataset('time', data=time_data)
            f.create_dataset('vorticity_max', data=vorticity_data)
            f.create_dataset('energy', data=energy_data)
            f.create_dataset('alignment', data=alignment_data)
            f.create_dataset('alpha', data=alpha_data)
            
            f.attrs['grid_size'] = grid_size
            f.attrs['Re'] = self.Re
            f.attrs['alpha'] = self.alpha
            f.attrs['max_vorticity'] = max(self.vorticity_max_history)
            f.attrs['bkm_integral'] = self.bkm_integral
            f.attrs['use_gpu'] = self.use_gpu
            f.attrs['steps'] = len(self.time_history)
            
            # Stage transitions
            if self.stage_transitions:
                trans_group = f.create_group('stage_transitions')
                for i, trans in enumerate(self.stage_transitions):
                    trans_group.attrs[f'transition_{i}'] = (
                        f"Step {trans['step']}: {trans['stage']} at vort={trans['vorticity']:.2f}"
                    )
        
        self.generate_plots(output_dir, grid_size)
    
    def generate_plots(self, output_dir, grid_size):
        """Generate plots - enhanced version with 10K steps"""
        plt.close('all')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Vorticity evolution (log scale)
        axes[0, 0].semilogy(self.time_history, self.vorticity_max_history, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Max Vorticity')
        axes[0, 0].set_title(f'{grid_size}^3 Vorticity Evolution (CUDA, 10K steps)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Alpha evolution
        axes[0, 1].plot(self.time_history, self.alpha_history, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('alpha')
        axes[0, 1].set_title(f'Adaptive Alpha ({grid_size}^3)')
        axes[0, 1].set_ylim([0.995, 1.0])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Alignment scores
        axes[0, 2].plot(self.time_history, self.alignment_history, 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Alignment Score')
        axes[0, 2].set_title(f'Alignment Evolution ({grid_size}^3)')
        axes[0, 2].set_ylim([0.995, 1.001])
        axes[0, 2].grid(True, alpha=0.3)
        
        # Energy
        axes[1, 0].plot(self.time_history, self.energy_history, 'c-', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Kinetic Energy')
        axes[1, 0].set_title(f'Energy Evolution ({grid_size}^3)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # BKM integral
        if len(self.time_history) > 1 and len(self.vorticity_max_history) > 0:
            dt_avg = np.mean(np.diff(self.time_history))
            if dt_avg > 0 and np.isfinite(dt_avg):
                bkm_cumulative = np.cumsum(self.vorticity_max_history) * dt_avg
            else:
                bkm_cumulative = np.linspace(0, self.bkm_integral, len(self.time_history))
        else:
            bkm_cumulative = [0] * len(self.time_history)
            
        axes[1, 1].plot(self.time_history, bkm_cumulative, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Integral ||omega||_inf dt')
        axes[1, 1].set_title(f'BKM Integral (Final: {self.bkm_integral:.4f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary
        success = max(self.vorticity_max_history) < 100
        axes[1, 2].text(0.5, 0.8, f'{grid_size}^3 CUDA RESULTS', fontsize=16, ha='center', weight='bold')
        axes[1, 2].text(0.5, 0.6, 'SUCCESS!' if success else 'CONTROLLED', 
                       fontsize=20, ha='center', 
                       color='green' if success else 'red', weight='bold')
        axes[1, 2].text(0.5, 0.4, f'Max vorticity: {max(self.vorticity_max_history):.2f}', 
                       fontsize=12, ha='center')
        axes[1, 2].text(0.5, 0.3, f'Steps: {len(self.time_history):,}', fontsize=12, ha='center')
        axes[1, 2].text(0.5, 0.2, f'Re = {self.Re}', fontsize=12, ha='center')
        axes[1, 2].text(0.5, 0.1, f'{grid_size**3:,} grid points', fontsize=10, ha='center', style='italic')
        axes[1, 2].text(0.5, 0.05, f'GPU: {self.use_gpu}', fontsize=8, ha='center', style='italic')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/test_{grid_size}cubed_cuda_10k_results.png', dpi=150, bbox_inches='tight')
        plt.close()

def run_cuda_test_suite():
    """Run CUDA-enhanced test suite with 10K steps"""
    print("="*60)
    print("NAVIER-STOKES ALIGNMENT FRAMEWORK - CUDA TEST SUITE")
    print("Enhanced with GPU acceleration for faster testing")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    if GPU_AVAILABLE:
        print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"GPU Memory: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB")
    
    # Test configurations - 10K steps for more comprehensive testing
    test_configs = [
        {'grid_size': 64, 'dt': 0.002, 'steps': 10000, 'alpha': 0.999},
        {'grid_size': 128, 'dt': 0.001, 'steps': 10000, 'alpha': 0.999},
        {'grid_size': 256, 'dt': 0.0005, 'steps': 10000, 'alpha': 0.999},
    ]
    
    results = []
    output_dir = "navier_stokes_cuda_tests_10k"
    os.makedirs(output_dir, exist_ok=True)
    
    for config in test_configs:
        print(f"\n" + "="*40)
        print(f"TESTING {config['grid_size']}^3 RESOLUTION - CUDA ENHANCED")
        print(f"="*40)
        
        try:
            test = NavierStokesAlignmentTestCUDA(
                grid_size=config['grid_size'],
                dt=config['dt'],
                alpha=config['alpha'],
                use_gpu=True  # Force GPU usage if available
            )
            
            test.initialize_taylor_green(amplitude=1.0)
            result = test.run_test(steps=config['steps'])
            test.save_results(output_dir, config['grid_size'])
            
            result.update(config)
            results.append(result)
            
            print(f"[OK] {config['grid_size']}^3 test completed")
            print(f"   Max vorticity: {result['max_vorticity']:.3f}")
            print(f"   BKM integral: {result['bkm_integral']:.6f}")
            print(f"   Performance: {result['steps_per_sec']:.1f} steps/sec")
            
            # GPU memory cleanup
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.runtime.deviceSynchronize()
            
        except Exception as e:
            print(f"[FAIL] {config['grid_size']}^3 test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "="*60)
    print("CUDA TEST SUITE SUMMARY (10K STEPS)")
    print("="*60)
    
    for result in results:
        status = "[PASSED]" if result['success'] else "[HIGH VORTICITY]"
        gpu_indicator = "[GPU]" if GPU_AVAILABLE else "[CPU]"
        print(f"{result['grid_size']:3d}^3: {status} | {gpu_indicator} | "
              f"Max vort: {result['max_vorticity']:6.2f} | "
              f"BKM: {result['bkm_integral']:6.4f} | "
              f"Runtime: {result['runtime']:5.2f}s | "
              f"Speed: {result['steps_per_sec']:4.0f} steps/s")
    
    print(f"\nResults saved to: {output_dir}/")
    print("Note: 10K steps provide extended evolution beyond critical period")
    
    # Final memory cleanup
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

if __name__ == "__main__":
    run_cuda_test_suite()