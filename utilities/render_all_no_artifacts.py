import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
from scipy.fft import fftn
from numpy.fft import fftfreq

# Force matplotlib to use a non-interactive backend
matplotlib.use('Agg')

# === USER CONFIG ===
output_dir = "pure_images"
os.makedirs(output_dir, exist_ok=True)

def compute_vorticity(u, v, w, spacing=(1, 1, 1)):
    """Compute vorticity magnitude"""
    dudx, dudy, dudz = np.gradient(u, *spacing)
    dvdx, dvdy, dvdz = np.gradient(v, *spacing)
    dwdx, dwdy, dwdz = np.gradient(w, *spacing)
    omega_x = dwdy - dvdz
    omega_y = dudz - dwdx
    omega_z = dvdx - dudy
    return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

def save_pure_image(data, output_path, cmap='plasma'):
    """Save pure image with zero matplotlib elements"""
    # Nuclear cleanup
    plt.close('all')
    plt.clf()
    plt.cla()
    
    # Clear matplotlib registries
    if hasattr(plt, '_pylab_helpers'):
        plt._pylab_helpers.Gcf.destroy_all()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    try:
        # Create figure with zero margins
        fig = plt.figure(figsize=(8, 8), dpi=150, facecolor='black')
        fig.clear()
        
        # Create axes that fill entire figure
        ax = fig.add_axes([0, 0, 1, 1])
        ax.clear()
        
        # JUST THE IMAGE - NOTHING ELSE
        im = ax.imshow(data, cmap=cmap, origin='lower', aspect='equal')
        
        # Remove ALL axes elements
        ax.axis('off')  # No axes, no ticks, no frame, NOTHING
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Pure black background
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Save with tight cropping
        fig.savefig(output_path, 
                   dpi=150,
                   bbox_inches='tight',
                   facecolor='black',
                   edgecolor='none',
                   pad_inches=0,
                   transparent=False,
                   format='png')
        
        print(f"[OK] Saved: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Error saving {output_path}: {e}")
        
    finally:
        # Complete destruction
        try:
            ax.clear()
            fig.clear()
            plt.close(fig)
        except:
            pass
        plt.close('all')
        plt.clf()
        plt.cla()
        
        # Final cleanup
        if hasattr(plt, '_pylab_helpers'):
            plt._pylab_helpers.Gcf.destroy_all()
        gc.collect()

def radial_spectrum(field, bins=100):
    """Compute radial energy spectrum"""
    fhat = fftn(field)
    E = np.abs(fhat)**2
    nx, ny, nz = field.shape
    kx, ky, kz = fftfreq(nx), fftfreq(ny), fftfreq(nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    kr = np.sqrt(KX**2 + KY**2 + KZ**2).flatten()
    E_flat = E.flatten()
    hist, edges = np.histogram(kr, bins=bins, weights=E_flat)
    k_vals = 0.5 * (edges[1:] + edges[:-1])
    return k_vals, hist

def save_spectrum_pure(k, E_k, output_path):
    """Save spectrum plot with minimal styling"""
    plt.close('all')
    
    try:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        
        ax.loglog(k, E_k + 1e-16, 'b-', linewidth=2)
        ax.set_xlabel("Wavenumber k", fontsize=14)
        ax.set_ylabel("E(k)", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Clean styling
        ax.tick_params(labelsize=12)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"[OK] Saved spectrum: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Error saving spectrum {output_path}: {e}")
        
    finally:
        plt.close('all')

# Main processing
print("=== Pure Image Renderer ===")
print("Generating clean visualizations with no text or artifacts")

# Find checkpoint files
files = sorted(glob.glob("checkpoint_step_*.h5"))
if not files:
    print("No checkpoint files found! Looking for checkpoint_step_*.h5")
    exit(1)

def get_step(fname): 
    return int(fname.split("_")[-1].split(".")[0])

files = sorted(files, key=get_step)
print(f"Found {len(files)} checkpoint files")

# Create output directories
planes = ["xy", "xz", "yz"]
for plane in planes:
    os.makedirs(f"{output_dir}/vorticity_{plane}", exist_ok=True)
    os.makedirs(f"{output_dir}/velocity_{plane}", exist_ok=True)
os.makedirs(f"{output_dir}/spectra", exist_ok=True)

# Process each checkpoint
for fname in files:
    step = get_step(fname)
    print(f"\nStep {step}: Processing {fname}")
    
    try:
        with h5py.File(fname, 'r') as f:
            u = np.array(f['u'])
            v = np.array(f['v'])
            w = np.array(f['w'])
        
        print(f"  Grid shape: {u.shape}")
        
        # Compute derived fields
        vel_mag = np.sqrt(u**2 + v**2 + w**2)
        vort_mag = compute_vorticity(u, v, w)
        mid = u.shape[0] // 2
        
        print(f"  Max vorticity: {np.max(vort_mag):.3f}")
        print(f"  Max velocity: {np.max(vel_mag):.3f}")
        
        # Save vorticity slices
        print("  Saving vorticity images...")
        for plane in planes:
            if plane == "xy":
                data = vort_mag[:, :, mid]
            elif plane == "xz":
                data = vort_mag[:, mid, :]
            elif plane == "yz":
                data = vort_mag[mid, :, :]
            
            output_path = f"{output_dir}/vorticity_{plane}/step_{step:06d}.png"
            save_pure_image(data, output_path, cmap='plasma')
        
        # Save velocity slices
        print("  Saving velocity images...")
        for plane in planes:
            if plane == "xy":
                data = vel_mag[:, :, mid]
            elif plane == "xz":
                data = vel_mag[:, mid, :]
            elif plane == "yz":
                data = vel_mag[mid, :, :]
            
            output_path = f"{output_dir}/velocity_{plane}/step_{step:06d}.png"
            save_pure_image(data, output_path, cmap='viridis')
        
        # Save spectrum
        print("  Computing spectrum...")
        k, E_k = radial_spectrum(vel_mag)
        spectrum_path = f"{output_dir}/spectra/step_{step:06d}.png"
        save_spectrum_pure(k, E_k, spectrum_path)
        
    except Exception as e:
        print(f"  ERROR processing {fname}: {e}")
        continue

print(f"\n[COMPLETE] Pure image rendering complete!")
print(f"Output directory: {output_dir}")
print("Generated clean visualizations with:")
print("  - Pure vorticity images (plasma colormap)")
print("  - Pure velocity images (viridis colormap)")  
print("  - Clean energy spectra")
print("  - No text, labels, axes, or artifacts!")