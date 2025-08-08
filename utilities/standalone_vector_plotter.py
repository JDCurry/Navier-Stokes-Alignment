"""
Standalone Vector Field Plotter
Runs independently to avoid matplotlib state contamination
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import os
import sys
import argparse

def compute_vorticity(u, v, w, spacing=(1, 1, 1)):
    """Compute vorticity magnitude"""
    dudx, dudy, dudz = np.gradient(u, *spacing)
    dvdx, dvdy, dvdz = np.gradient(v, *spacing)
    dwdx, dwdy, dwdz = np.gradient(w, *spacing)
    omega_x = dwdy - dvdz
    omega_y = dudz - dwdx
    omega_z = dvdx - dudy
    return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

def create_vector_plot(vort, u, v, w, output_path, plane, mid):
    """Create pure vorticity image with ZERO extra elements"""
    # Nuclear matplotlib reset
    plt.close('all')
    plt.clf()
    plt.cla()
    
    # Clear matplotlib's internal registries
    import matplotlib._pylab_helpers
    matplotlib._pylab_helpers.Gcf.destroy_all()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Create figure with explicit parameters
    fig = plt.figure(figsize=(8, 8), dpi=150, facecolor='black')
    fig.clear()
    
    # Create axes with zero margins
    ax = fig.add_axes([0, 0, 1, 1])
    ax.clear()
    
    try:
        # JUST THE IMAGE - NO ARROWS, NO TEXT, NO ANYTHING
        if plane == "xy":
            im = ax.imshow(vort[:, :, mid], cmap='plasma', origin='lower', aspect='equal')
        elif plane == "xz":
            im = ax.imshow(vort[:, mid, :], cmap='plasma', origin='lower', aspect='equal')
        elif plane == "yz":
            im = ax.imshow(vort[mid, :, :], cmap='plasma', origin='lower', aspect='equal')
        
        # Remove EVERYTHING
        ax.axis('off')  # No axes, no ticks, no frame, NOTHING
        
        # Pure black background
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Save with minimal parameters
        fig.savefig(output_path, 
                   dpi=150, 
                   bbox_inches='tight',  # Let it crop tight
                   facecolor='black',
                   edgecolor='none',
                   pad_inches=0,
                   transparent=False,
                   format='png')
        
        print(f"[SUCCESS] Saved pure image: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Error creating {plane} image: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Complete destruction
        ax.clear()
        fig.clear()
        plt.close(fig)
        plt.close('all')
        plt.clf()
        plt.cla()
        
        # Clear matplotlib caches
        matplotlib._pylab_helpers.Gcf.destroy_all()
        gc.collect()

def process_checkpoint(checkpoint_file, output_dir, step):
    """Process a single checkpoint file"""
    print(f"Processing checkpoint: {checkpoint_file}")
    
    try:
        with h5py.File(checkpoint_file, 'r') as f:
            u = np.array(f['u'])
            v = np.array(f['v'])
            w = np.array(f['w'])
        
        # Compute vorticity
        vort_mag = compute_vorticity(u, v, w)
        mid = u.shape[0] // 2
        
        # Create output directories
        planes = ["xy", "xz", "yz"]
        for plane in planes:
            plane_dir = f"{output_dir}/vectors/{plane}"
            os.makedirs(plane_dir, exist_ok=True)
            
            # Create vector plot for this plane
            output_path = f"{plane_dir}/step_{step:06d}_{plane}.png"
            create_vector_plot(vort_mag, u, v, w, output_path, plane, mid)
        
    except Exception as e:
        print(f"Error processing {checkpoint_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Standalone Vector Field Plotter')
    parser.add_argument('--checkpoint', '-c', required=True, help='Checkpoint file to process')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory')
    parser.add_argument('--step', '-s', type=int, required=True, help='Step number')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    process_checkpoint(args.checkpoint, args.output_dir, args.step)
    print("Vector plotting complete!")

if __name__ == "__main__":
    main()