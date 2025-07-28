# ====================================================================
# CORRECTED PROFESSIONAL DATA AUGMENTATION STRATEGY
# ====================================================================
# This fixes the flipping issue and uses proper rotation angles
# with realistic distribution targeting

import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def add_gaussian_noise(sip, noise_level=0.15):
    """
    Add Gaussian noise with 15% of the maximum possible value (4000).
    This ensures the model can distinguish between real and augmented data.
    """
    max_possible_val = 4000  # Maximum possible sensor value
    std_dev = noise_level * max_possible_val
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
    """
    Return a limited number of rotated versions of sip.
    Using realistic rotation angles for chest-mounted sensor study.
    These angles represent realistic body movements during drinking.
    """
    # Realistic rotation angles for chest-mounted sensor
    # Small movements: 5°, 10°, 15° (realistic body adjustments)
    # Moderate movements: 20°, 30°, 45° (leaning, turning)
    # Large movements: 315°, 330°, 340°, 350°, 355° (counter-rotations)
    realistic_rotation_angles = [5, 355, 10, 350, 20, 340, 30, 330, 45, 315]
    
    # Randomly select angles for diversity
    selected_angles = np.random.choice(realistic_rotation_angles, 
                                     size=min(num_rotations, len(realistic_rotation_angles)), 
                                     replace=False)
    
    return [rotate_sip(sip, angle) for angle in selected_angles]

def analyze_volume_distribution_corrected(y_values, bin_width=5):
    """
    SMART SMOOTHING distribution analysis to eliminate ALL drops and create uniform distribution.
    Uses intelligent targeting to avoid over-augmentation.
    """
    bins = np.arange(0, max(y_values) + bin_width, bin_width)
    hist, bin_edges = np.histogram(y_values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate statistics
    mean_samples = np.mean(hist[hist > 0])
    median_samples = np.median(hist[hist > 0])
    
    # SMART target: aim for uniform distribution around mean
    target_samples = int(mean_samples * 1.2)  # 20% above mean for safety
    
    # SMART augmentation needs calculation
    augmentation_needs = {}
    
    # First pass: identify ALL problematic drops
    problematic_drops = []
    for i in range(1, len(hist)):
        if hist[i] > 0 and hist[i-1] > 0:
            drop_ratio = hist[i] / hist[i-1]
            if drop_ratio < 0.8:  # More than 20% drop (very sensitive)
                problematic_drops.append((i, drop_ratio))
    
    # Second pass: calculate SMART augmentation needs
    for i, (center, count) in enumerate(zip(bin_centers, hist)):
        if count == 0:  # Empty bins
            augmentation_needs[center] = int(target_samples * 0.8)
        elif count < target_samples * 0.5:  # Severely underrepresented
            augmentation_needs[center] = int(target_samples - count)
        elif count < target_samples * 0.8:  # Moderately underrepresented
            augmentation_needs[center] = int((target_samples - count) * 0.7)
        elif count < target_samples * 1.1:  # Slightly underrepresented
            augmentation_needs[center] = int((target_samples - count) * 0.3)
        else:  # Well represented - minimal augmentation
            augmentation_needs[center] = 0
    
    # Third pass: SMART smoothing of problematic drops
    for drop_idx, drop_ratio in problematic_drops:
        if drop_idx < len(bin_centers):
            center = bin_centers[drop_idx]
            if center in augmentation_needs:
                # Calculate how much we need to smooth the drop
                current_need = augmentation_needs[center]
                # Aim to bring the drop up to 80% of the previous bin
                target_for_drop = int(hist[drop_idx-1] * 0.8)
                additional_need = max(0, target_for_drop - hist[drop_idx])
                augmentation_needs[center] = max(current_need, additional_need)
    
    # Fourth pass: prevent over-augmentation of already high bins
    for center, needed in list(augmentation_needs.items()):
        bin_idx = np.where(bin_centers == center)[0][0]
        current_count = hist[bin_idx]
        if current_count > target_samples * 1.5:  # If already over-augmented
            augmentation_needs[center] = 0  # Stop augmenting this bin
    
    return augmentation_needs, bin_centers, hist, target_samples

def determine_augmentation_strategy_corrected(dV, augmentation_needs, bin_width=5):
    """
    Corrected augmentation strategy with proper flipping (only once) and better rotation angles.
    """
    bin_center = int(dV // bin_width) * bin_width + bin_width / 2
    
    # SMART BALANCED approach: intelligent targeting to avoid over-augmentation
    if dV <= 20:  # Low DV ranges - moderate augmentation
        if bin_center in augmentation_needs:
            needed = augmentation_needs[bin_center]
            if needed > 15:
                return {
                    'rotations': 3,  # Balanced rotations
                    'flips': 1,  # Only flip once!
                    'noise_copies': 1,
                    'noise_levels': [0.15]  # Single noise level
                }
            else:
                return {
                    'rotations': 2,
                    'flips': 1,  # Only flip once!
                    'noise_copies': 0,
                    'noise_levels': []
                }
        else:
            return {
                'rotations': 1,  # Minimal rotation
                'flips': 0,
                'noise_copies': 0,
                'noise_levels': []
            }
    
    elif dV > 100:  # High DV ranges - moderate augmentation
        if bin_center in augmentation_needs:
            needed = augmentation_needs[bin_center]
            return {
                'rotations': 3,  # Balanced rotations
                'flips': 1,  # Only flip once!
                'noise_copies': 1,
                'noise_levels': [0.15]  # Single noise level
            }
        else:
            return {
                'rotations': 2,
                'flips': 1,  # Only flip once!
                'noise_copies': 0,
                'noise_levels': []
            }
    
    elif dV > 60:  # Medium-high DV ranges
        if bin_center in augmentation_needs:
            needed = augmentation_needs[bin_center]
            if needed > 10:
                return {
                    'rotations': 3,
                    'flips': 1,  # Only flip once!
                    'noise_copies': 1,
                    'noise_levels': [0.15]
                }
            else:
                return {
                    'rotations': 2,
                    'flips': 1,  # Only flip once!
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
    
    else:  # Medium DV ranges (20-60) - SMART targeting of drops
        if bin_center in augmentation_needs:
            needed = augmentation_needs[bin_center]
            if needed > 15:
                return {
                    'rotations': 3,  # Balanced rotations for problematic areas
                    'flips': 1,  # Only flip once!
                    'noise_copies': 1,
                    'noise_levels': [0.15]
                }
            else:
                return {
                    'rotations': 2,
                    'flips': 1,  # Only flip once!
                    'noise_copies': 0,
                    'noise_levels': []
                }
        else:
            return {
                'rotations': 1,  # Minimal rotation
                'flips': 0,
                'noise_copies': 0,
                'noise_levels': []
            }

def apply_corrected_professional_augmentation(df, feature_cols, seq_keys, label_col):
    """
    Corrected professional data augmentation with proper flipping and rotation.
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
    
    # Corrected distribution analysis
    original_volumes = [data[1] for data in original_data]
    bin_width = 5  # 5mL bins for better detail as requested
    augmentation_needs, bin_centers, hist, target_samples = analyze_volume_distribution_corrected(original_volumes, bin_width)
    
    print("=== CORRECTED PROFESSIONAL Volume Distribution Analysis ===")
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
    
    # Second pass: apply corrected professional augmentation
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
        
        # Determine corrected augmentation strategy
        strategy = determine_augmentation_strategy_corrected(dV, augmentation_needs, bin_width)
        
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
        
        # Apply flips (ONLY ONCE per sample!)
        if strategy['flips'] > 0:
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
    
    print(f"\n=== CORRECTED PROFESSIONAL Dataset Statistics ===")
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
    
    print("\nFinal distribution after corrected professional augmentation:")
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

def plot_corrected_comparison(original_volumes, final_volumes, bin_width=5):
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
    ax2.set_title('Final Volume Distribution (After Corrected Augmentation)')
    ax2.set_xlabel('Volume (mL)')
    ax2.set_ylabel('Number of Samples')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution function
def run_corrected_professional_augmentation(df, feature_cols, seq_keys, label_col):
    """
    Run the complete corrected professional augmentation pipeline.
    """
    print("Starting CORRECTED PROFESSIONAL Data Augmentation...")
    print("=" * 60)
    
    X, y, is_original, y_before_aug = apply_corrected_professional_augmentation(
        df, feature_cols, seq_keys, label_col
    )
    
    # Plot comparison
    original_volumes = y_before_aug
    final_volumes = y
    plot_corrected_comparison(original_volumes, final_volumes)
    
    return X, y, is_original, y_before_aug 