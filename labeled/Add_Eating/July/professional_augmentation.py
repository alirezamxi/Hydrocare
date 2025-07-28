# ====================================================================
# PROFESSIONAL DATA AUGMENTATION STRATEGY
# ====================================================================
# This implements proper data science techniques for balanced augmentation
# with appropriate noise levels (10-20%) and realistic distribution targeting

import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def add_gaussian_noise(sip, noise_level=0.15):
    """
    Add Gaussian noise with 15% intensity (as recommended by professor).
    This ensures the model can distinguish between real and augmented data.
    """
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
    rotation_angles = [15, 345, 30, 330]  # 4 angles
    selected_angles = np.random.choice(rotation_angles, size=min(num_rotations, len(rotation_angles)), replace=False)
    return [rotate_sip(sip, angle) for angle in selected_angles]

def analyze_volume_distribution_professional(y_values, bin_width=10):
    """
    Professional distribution analysis with realistic targeting.
    Targets a balanced distribution without over-augmentation.
    """
    bins = np.arange(0, max(y_values) + bin_width, bin_width)
    hist, bin_edges = np.histogram(y_values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate statistics
    mean_samples = np.mean(hist[hist > 0])
    median_samples = np.median(hist[hist > 0])
    
    # Target: aim for a reasonable balance without over-augmentation
    # Use 75% of the peak as target to avoid over-augmentation
    peak_samples = np.max(hist)
    target_samples = int(peak_samples * 0.75)
    
    # Professional augmentation needs calculation
    augmentation_needs = {}
    for i, (center, count) in enumerate(zip(bin_centers, hist)):
        if count == 0:  # Empty bins need significant augmentation
            augmentation_needs[center] = int(target_samples * 0.6)
        elif count < target_samples * 0.3:  # Severely underrepresented
            augmentation_needs[center] = int(target_samples - count)
        elif count < target_samples * 0.6:  # Moderately underrepresented
            augmentation_needs[center] = int((target_samples - count) * 0.7)
        elif count < target_samples * 0.8:  # Slightly underrepresented
            augmentation_needs[center] = int((target_samples - count) * 0.4)
    
    return augmentation_needs, bin_centers, hist, target_samples

def determine_augmentation_strategy_professional(dV, augmentation_needs, bin_width=10):
    """
    Professional augmentation strategy with appropriate noise levels and realistic approach.
    """
    bin_center = int(dV // bin_width) * bin_width + bin_width / 2
    
    # Professional approach: different strategies for different volume ranges
    if dV <= 20:  # Low DV ranges - minimal augmentation with rotation
        if bin_center in augmentation_needs:
            needed = augmentation_needs[bin_center]
            if needed > 15:
                return {
                    'rotations': 2,
                    'flips': 1,
                    'noise_copies': 1,
                    'noise_levels': [0.15]  # 15% noise as recommended
                }
            else:
                return {
                    'rotations': 1,
                    'flips': 0,
                    'noise_copies': 0,
                    'noise_levels': []
                }
        else:
            return {
                'rotations': 1,  # Always add rotation for robustness
                'flips': 0,
                'noise_copies': 0,
                'noise_levels': []
            }
    
    elif dV > 100:  # High DV ranges - significant augmentation needed
        if bin_center in augmentation_needs:
            needed = augmentation_needs[bin_center]
            return {
                'rotations': 3,
                'flips': 2,
                'noise_copies': 3,
                'noise_levels': [0.12, 0.15, 0.18]  # Multiple noise levels
            }
        else:
            return {
                'rotations': 2,
                'flips': 1,
                'noise_copies': 2,
                'noise_levels': [0.12, 0.15]
            }
    
    elif dV > 60:  # Medium-high DV ranges
        if bin_center in augmentation_needs:
            needed = augmentation_needs[bin_center]
            if needed > 20:
                return {
                    'rotations': 3,
                    'flips': 2,
                    'noise_copies': 2,
                    'noise_levels': [0.12, 0.15]
                }
            else:
                return {
                    'rotations': 2,
                    'flips': 1,
                    'noise_copies': 1,
                    'noise_levels': [0.15]
                }
        else:
            return {
                'rotations': 1,
                'flips': 0,
                'noise_copies': 0,
                'noise_levels': []
            }
    
    else:  # Medium DV ranges (20-60)
        if bin_center in augmentation_needs:
            needed = augmentation_needs[bin_center]
            if needed > 25:
                return {
                    'rotations': 2,
                    'flips': 1,
                    'noise_copies': 1,
                    'noise_levels': [0.15]
                }
            else:
                return {
                    'rotations': 1,
                    'flips': 1,
                    'noise_copies': 0,
                    'noise_levels': []
                }
        else:
            return {
                'rotations': 1,
                'flips': 0,
                'noise_copies': 0,
                'noise_levels': []
            }

def apply_professional_augmentation(df, feature_cols, seq_keys, label_col):
    """
    Professional data augmentation with proper distribution balancing.
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
    
    # Professional distribution analysis
    original_volumes = [data[1] for data in original_data]
    bin_width = 10
    augmentation_needs, bin_centers, hist, target_samples = analyze_volume_distribution_professional(original_volumes, bin_width)
    
    print("=== PROFESSIONAL Volume Distribution Analysis ===")
    print(f"Original samples: {len(original_data)}")
    print(f"Target samples per bin: {target_samples}")
    print(f"Peak samples: {np.max(hist)}")
    print(f"Mean samples per bin: {np.mean(hist[hist > 0]):.1f}")
    print(f"Median samples per bin: {np.median(hist[hist > 0]):.1f}")
    
    print("\nVolume bins and sample counts:")
    for center, count in zip(bin_centers, hist):
        if count > 0:
            print(f"  {center-bin_width/2:.1f}-{center+bin_width/2:.1f} mL: {count} samples")
    
    print("\nAugmentation needs:")
    for center, needed in augmentation_needs.items():
        print(f"  {center-bin_width/2:.1f}-{center+bin_width/2:.1f} mL: need {needed} more samples")
    
    # Second pass: apply professional augmentation
    augmentation_count = 0
    low_dv_augmented = 0
    high_dv_augmented = 0
    medium_dv_augmented = 0
    
    for sip_frames, dV in original_data:
        # Add original sample
        X.append(sip_frames)
        y.append(dV)
        y_before_aug.append(dV)
        is_original.append(True)
        
        # Determine professional augmentation strategy
        strategy = determine_augmentation_strategy_professional(dV, augmentation_needs, bin_width)
        
        # Track augmentation by volume range
        if dV <= 20:
            range_type = "low"
        elif dV > 100:
            range_type = "high"
        else:
            range_type = "medium"
        
        # Apply rotations
        if strategy['rotations'] > 0:
            rotations = get_rotations(sip_frames, strategy['rotations'])
            for rotated in rotations:
                X.append(rotated)
                y.append(dV)
                is_original.append(False)
                augmentation_count += 1
                if range_type == "low":
                    low_dv_augmented += 1
                elif range_type == "high":
                    high_dv_augmented += 1
                else:
                    medium_dv_augmented += 1
        
        # Apply flips
        for _ in range(strategy['flips']):
            flipped = flip_horizontal(sip_frames)
            X.append(flipped)
            y.append(dV)
            is_original.append(False)
            augmentation_count += 1
            if range_type == "low":
                low_dv_augmented += 1
            elif range_type == "high":
                high_dv_augmented += 1
            else:
                medium_dv_augmented += 1
        
        # Apply noise with appropriate levels
        for _ in range(strategy['noise_copies']):
            for noise_level in strategy['noise_levels']:
                noisy = add_gaussian_noise(sip_frames, noise_level)
                X.append(noisy)
                y.append(dV)
                is_original.append(False)
                augmentation_count += 1
                if range_type == "low":
                    low_dv_augmented += 1
                elif range_type == "high":
                    high_dv_augmented += 1
                else:
                    medium_dv_augmented += 1
    
    # Final conversion to array
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=np.float32)
    y_before_aug = np.array(y_before_aug, dtype=np.float32)
    is_original = np.array(is_original, dtype=bool)
    
    print(f"\n=== PROFESSIONAL Dataset Statistics ===")
    print(f"Total sips kept (including augmented): {len(X)}")
    print(f"Zero-volume skipped: {skip_count}")
    print(f"Label stats → min: {min(y):.1f}, max: {max(y):.1f}")
    print(f"Original sips: {np.sum(is_original)}, Augmented sips: {np.sum(~is_original)}")
    print(f"Augmentation ratio: {np.sum(~is_original) / np.sum(is_original):.2f}x")
    print(f"Total augmentations applied: {augmentation_count}")
    
    print(f"\n=== Augmentation by Volume Range ===")
    print(f"Low DV (≤20mL) augmentations: {low_dv_augmented}")
    print(f"Medium DV (21-100mL) augmentations: {medium_dv_augmented}")
    print(f"High DV (>100mL) augmentations: {high_dv_augmented}")
    
    # Verify the new distribution is more balanced
    final_volumes = y
    final_hist, _ = np.histogram(final_volumes, bins=bin_centers)
    
    print("\nFinal distribution after professional augmentation:")
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
    
    # Additional metrics
    original_cv = original_std / np.mean(hist[hist > 0])  # Coefficient of variation
    final_cv = final_std / np.mean(final_hist[final_hist > 0])
    cv_improvement = (original_cv - final_cv) / original_cv * 100
    
    print(f"Original CV: {original_cv:.3f}")
    print(f"Final CV: {final_cv:.3f}")
    print(f"CV improvement: {cv_improvement:.1f}%")
    
    return X, y, is_original, y_before_aug

def plot_professional_comparison(original_volumes, final_volumes, bin_width=10):
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
    ax2.set_title('Final Volume Distribution (After Professional Augmentation)')
    ax2.set_xlabel('Volume (mL)')
    ax2.set_ylabel('Number of Samples')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution function
def run_professional_augmentation(df, feature_cols, seq_keys, label_col):
    """
    Run the complete professional augmentation pipeline.
    """
    print("Starting PROFESSIONAL Data Augmentation...")
    print("=" * 60)
    
    X, y, is_original, y_before_aug = apply_professional_augmentation(
        df, feature_cols, seq_keys, label_col
    )
    
    # Plot comparison
    original_volumes = y_before_aug
    final_volumes = y
    plot_professional_comparison(original_volumes, final_volumes)
    
    return X, y, is_original, y_before_aug 