# ====================================================================
# ADVANCED BALANCED DATA AUGMENTATION FOR REGRESSION
# ====================================================================
# This implements advanced data science techniques for handling
# imbalanced datasets in regression problems

import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def add_gaussian_noise(sip, noise_level=0.03):
    """Add Gaussian noise with adaptive intensity based on signal strength."""
    max_val = np.max(sip)
    std_dev = noise_level * max_val
    noise = np.random.normal(0, std_dev, sip.shape)
    noisy_sip = sip + noise
    return np.clip(noisy_sip, a_min=0.0, a_max=None)

def flip_horizontal(sip):
    """Flip 8x8 frames horizontally for all time steps in sip (T, 64)."""
    flipped = []
    for frame in sip:
        frame_8x8 = frame.reshape(8, 8)
        flipped_frame = np.fliplr(frame_8x8).flatten()
        flipped.append(flipped_frame)
    return np.array(flipped, dtype=np.float32)

def rotate_frame(frame, angle):
    """Rotate a single 8x8 frame by a given angle (in degrees)."""
    frame_8x8 = frame.reshape(8, 8)
    rotated = rotate(frame_8x8, angle, reshape=False, order=1, mode='nearest')
    return rotated.flatten()

def rotate_sip(sip, angle):
    """Rotate all frames in a sip (T, 64) by a given angle (in degrees)."""
    return np.array([rotate_frame(frame, angle) for frame in sip], dtype=np.float32)

def get_rotations(sip, num_rotations=2):
    """Return a limited number of rotated versions of sip."""
    rotation_angles = [15, 345, 30, 330]  # Reduced to 4 angles
    selected_angles = np.random.choice(rotation_angles, size=min(num_rotations, len(rotation_angles)), replace=False)
    return [rotate_sip(sip, angle) for angle in selected_angles]

def analyze_volume_distribution_advanced(y_values, bin_width=10):
    """
    Advanced distribution analysis with better binning strategy.
    Uses adaptive binning for better representation.
    """
    # Use smaller bins for better granularity
    bins = np.arange(0, max(y_values) + bin_width, bin_width)
    hist, bin_edges = np.histogram(y_values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate statistics for better targeting
    mean_samples = np.mean(hist[hist > 0])  # Mean of non-empty bins
    median_samples = np.median(hist[hist > 0])  # Median of non-empty bins
    
    # Target: aim for median level, not maximum
    target_samples = min(median_samples * 1.5, mean_samples * 1.2)
    
    # More sophisticated augmentation needs calculation
    augmentation_needs = {}
    for i, (center, count) in enumerate(zip(bin_centers, hist)):
        if count == 0:  # Empty bins need significant augmentation
            augmentation_needs[center] = int(target_samples * 0.8)
        elif count < target_samples * 0.3:  # Severely underrepresented
            augmentation_needs[center] = int(target_samples - count)
        elif count < target_samples * 0.6:  # Moderately underrepresented
            augmentation_needs[center] = int((target_samples - count) * 0.7)
        elif count < target_samples * 0.8:  # Slightly underrepresented
            augmentation_needs[center] = int((target_samples - count) * 0.3)
    
    return augmentation_needs, bin_centers, hist, target_samples

def determine_augmentation_strategy_advanced(dV, augmentation_needs, bin_width=10):
    """
    Advanced augmentation strategy with better control over augmentation intensity.
    """
    bin_center = int(dV // bin_width) * bin_width + bin_width / 2
    
    if bin_center in augmentation_needs:
        needed = augmentation_needs[bin_center]
        
        # More conservative augmentation strategy
        if needed > 50:  # Severely underrepresented
            return {
                'rotations': 3,
                'flips': 2,
                'noise_copies': 2,
                'noise_levels': [0.02, 0.04]
            }
        elif needed > 25:  # Moderately underrepresented
            return {
                'rotations': 2,
                'flips': 1,
                'noise_copies': 1,
                'noise_levels': [0.03]
            }
        elif needed > 10:  # Slightly underrepresented
            return {
                'rotations': 1,
                'flips': 1,
                'noise_copies': 0,
                'noise_levels': []
            }
        else:  # Minimal need
            return {
                'rotations': 1,
                'flips': 0,
                'noise_copies': 0,
                'noise_levels': []
            }
    else:
        # Well-represented volumes get minimal augmentation
        return {
            'rotations': 0,
            'flips': 0,
            'noise_copies': 0,
            'noise_levels': []
        }

def apply_advanced_balanced_augmentation(df, feature_cols, seq_keys, label_col):
    """
    Advanced balanced data augmentation with better distribution control.
    
    This approach:
    1. Uses adaptive binning
    2. Targets median distribution level
    3. Applies conservative augmentation
    4. Maintains realistic data distribution
    """
    
    # Filter to keep only annotated frames
    df = df[df["Label"] == 1]
    
    # Initialize data containers
    X, y, is_original = [], [], []
    y_before_aug = []
    skip_count = 0
    
    # First pass: collect original data
    original_data = []
    for _, g in df.groupby(seq_keys, sort=False):
        dV = float(g[label_col].iloc[0])
        if dV == 0.0:
            skip_count += 1
            continue
        
        sip_frames = g[feature_cols].to_numpy(np.float32)
        original_data.append((sip_frames, dV))
    
    # Advanced distribution analysis
    original_volumes = [data[1] for data in original_data]
    bin_width = 10
    augmentation_needs, bin_centers, hist, target_samples = analyze_volume_distribution_advanced(original_volumes, bin_width)
    
    print("=== ADVANCED Volume Distribution Analysis ===")
    print(f"Original samples: {len(original_data)}")
    print(f"Target samples per bin: {target_samples:.1f}")
    print(f"Mean samples per bin: {np.mean(hist[hist > 0]):.1f}")
    print(f"Median samples per bin: {np.median(hist[hist > 0]):.1f}")
    
    print("\nVolume bins and sample counts:")
    for center, count in zip(bin_centers, hist):
        if count > 0:
            print(f"  {center-bin_width/2:.1f}-{center+bin_width/2:.1f} mL: {count} samples")
    
    print("\nAugmentation needs:")
    for center, needed in augmentation_needs.items():
        print(f"  {center-bin_width/2:.1f}-{center+bin_width/2:.1f} mL: need {needed} more samples")
    
    # Second pass: apply advanced balanced augmentation
    augmentation_count = 0
    for sip_frames, dV in original_data:
        # Add original sample
        X.append(sip_frames)
        y.append(dV)
        y_before_aug.append(dV)
        is_original.append(True)
        
        # Determine advanced augmentation strategy
        strategy = determine_augmentation_strategy_advanced(dV, augmentation_needs, bin_width)
        
        # Apply rotations (limited)
        if strategy['rotations'] > 0:
            rotations = get_rotations(sip_frames, strategy['rotations'])
            for rotated in rotations:
                X.append(rotated)
                y.append(dV)
                is_original.append(False)
                augmentation_count += 1
        
        # Apply flips (limited)
        for _ in range(strategy['flips']):
            flipped = flip_horizontal(sip_frames)
            X.append(flipped)
            y.append(dV)
            is_original.append(False)
            augmentation_count += 1
        
        # Apply noise (very limited)
        for _ in range(strategy['noise_copies']):
            for noise_level in strategy['noise_levels']:
                noisy = add_gaussian_noise(sip_frames, noise_level)
                X.append(noisy)
                y.append(dV)
                is_original.append(False)
                augmentation_count += 1
    
    # Final conversion to array
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=np.float32)
    y_before_aug = np.array(y_before_aug, dtype=np.float32)
    is_original = np.array(is_original, dtype=bool)
    
    print(f"\n=== ADVANCED Final Dataset Statistics ===")
    print(f"Total sips kept (including augmented): {len(X)}")
    print(f"Zero-volume skipped: {skip_count}")
    print(f"Label stats → min: {min(y):.1f}, max: {max(y):.1f}")
    print(f"Original sips: {np.sum(is_original)}, Augmented sips: {np.sum(~is_original)}")
    print(f"Augmentation ratio: {np.sum(~is_original) / np.sum(is_original):.2f}x")
    print(f"Total augmentations applied: {augmentation_count}")
    
    # Verify the new distribution is more balanced
    final_volumes = y
    final_hist, _ = np.histogram(final_volumes, bins=bin_centers)
    
    print("\nFinal distribution after advanced balanced augmentation:")
    for center, count in zip(bin_centers, final_hist):
        if count > 0:
            print(f"  {center-bin_width/2:.1f}-{center+bin_width/2:.1f} mL: {count} samples")
    
    # Calculate distribution improvement metrics
    original_std = np.std(hist[hist > 0])
    final_std = np.std(final_hist[final_hist > 0])
    improvement = (original_std - final_std) / original_std * 100
    
    print(f"\n=== Distribution Balance Metrics ===")
    print(f"Original distribution std: {original_std:.2f}")
    print(f"Final distribution std: {final_std:.2f}")
    print(f"Balance improvement: {improvement:.1f}%")
    
    return X, y, is_original, y_before_aug

def plot_distribution_comparison(original_volumes, final_volumes, bin_width=10):
    """Plot before and after distribution comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original distribution
    bins = np.arange(0, max(original_volumes) + bin_width, bin_width)
    ax1.hist(original_volumes, bins=bins, alpha=0.7, color='red', edgecolor='black')
    ax1.set_title('Original Volume Distribution')
    ax1.set_xlabel('Volume (mL)')
    ax1.set_ylabel('Number of Samples')
    ax1.grid(True, alpha=0.3)
    
    # Final distribution
    ax2.hist(final_volumes, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title('Final Volume Distribution (After Augmentation)')
    ax2.set_xlabel('Volume (mL)')
    ax2.set_ylabel('Number of Samples')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage function
def run_advanced_augmentation(df, feature_cols, seq_keys, label_col):
    """
    Run the complete advanced augmentation pipeline.
    """
    print("Starting Advanced Balanced Data Augmentation...")
    print("=" * 60)
    
    X, y, is_original, y_before_aug = apply_advanced_balanced_augmentation(
        df, feature_cols, seq_keys, label_col
    )
    
    # Plot comparison
    original_volumes = y_before_aug
    final_volumes = y
    plot_distribution_comparison(original_volumes, final_volumes)
    
    return X, y, is_original, y_before_aug 