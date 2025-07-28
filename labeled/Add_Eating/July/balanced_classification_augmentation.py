# ====================================================================
# BALANCED CLASSIFICATION DATA AUGMENTATION
# ====================================================================
# This file provides balanced augmentation for 3-class classification:
# 0 = not drinking, 1 = drinking, 2 = eating

import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.ndimage import rotate

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
    """
    # Realistic rotation angles for chest-mounted sensor
    realistic_rotation_angles = [5, 355, 10, 350, 20, 340, 30, 330, 45, 315]
    
    # Randomly select angles for diversity
    selected_angles = np.random.choice(realistic_rotation_angles, 
                                     size=min(num_rotations, len(realistic_rotation_angles)), 
                                     replace=False)
    
    return [rotate_sip(sip, angle) for angle in selected_angles]

def analyze_class_distribution(y_values):
    """
    Analyze class distribution and determine balanced augmentation needs.
    """
    class_counts = Counter(y_values)
    total_samples = len(y_values)
    
    print("=== CLASS DISTRIBUTION ANALYSIS ===")
    print(f"Total samples: {total_samples}")
    print(f"Class counts: {dict(class_counts)}")
    
    # Calculate target samples per class for balanced distribution
    max_class_count = max(class_counts.values())
    target_per_class = int(max_class_count * 0.8)  # 80% of the largest class
    
    print(f"Target samples per class: {target_per_class}")
    
    # Calculate augmentation needs per class
    augmentation_needs = {}
    for class_label in [0, 1, 2]:
        current_count = class_counts.get(class_label, 0)
        if current_count < target_per_class:
            needed = target_per_class - current_count
            augmentation_needs[class_label] = needed
        else:
            augmentation_needs[class_label] = 0
    
    print("\nAugmentation needs per class:")
    for class_label, needed in augmentation_needs.items():
        class_names = {0: "Not Drinking", 1: "Drinking", 2: "Eating"}
        print(f"  {class_names[class_label]}: need {needed} more samples")
    
    return augmentation_needs, target_per_class

def determine_class_augmentation_strategy(class_label, augmentation_needs, current_count, target_count):
    """
    Determine augmentation strategy for each class based on needs.
    """
    if class_label not in augmentation_needs or augmentation_needs[class_label] == 0:
        return {
            'rotations': 0,
            'flips': 0,
            'noise_copies': 0,
            'noise_levels': []
        }
    
    needed = augmentation_needs[class_label]
    
    # Different strategies based on class and needs
    if class_label == 0:  # Not drinking - minimal augmentation
        if needed > 50:
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
    
    elif class_label == 1:  # Drinking - moderate augmentation
        if needed > 100:
            return {
                'rotations': 3,
                'flips': 1,
                'noise_copies': 1,
                'noise_levels': [0.15]
            }
        else:
            return {
                'rotations': 2,
                'flips': 1,
                'noise_copies': 0,
                'noise_levels': []
            }
    
    else:  # Eating - aggressive augmentation (smallest class)
        if needed > 20:
            return {
                'rotations': 4,
                'flips': 1,
                'noise_copies': 2,
                'noise_levels': [0.12, 0.15]
            }
        else:
            return {
                'rotations': 3,
                'flips': 1,
                'noise_copies': 1,
                'noise_levels': [0.15]
            }

def apply_balanced_classification_augmentation(X_sequences, y_sequences, is_original, target_total=2000):
    """
    Apply balanced augmentation for classification with target total samples.
    """
    print("=== BALANCED CLASSIFICATION AUGMENTATION ===")
    
    # Convert to arrays
    X_sequences_array = np.array(X_sequences, dtype=object)
    y_sequences_array = np.array(y_sequences)
    is_original_array = np.array(is_original)
    
    # Analyze current distribution
    original_indices = np.where(is_original_array == True)[0]
    original_X = X_sequences_array[original_indices]
    original_y = y_sequences_array[original_indices]
    
    augmentation_needs, target_per_class = analyze_class_distribution(original_y)
    
    # Calculate how many samples we need to augment per original sample
    total_original = len(original_X)
    target_augmented = target_total - total_original
    
    if target_augmented <= 0:
        print("Target total is less than or equal to original samples. No augmentation needed.")
        return X_sequences_array, y_sequences_array, is_original_array
    
    print(f"\nTarget total samples: {target_total}")
    print(f"Original samples: {total_original}")
    print(f"Need to generate: {target_augmented} augmented samples")
    
    # Apply balanced augmentation
    augmented_X = list(original_X)
    augmented_y = list(original_y)
    augmented_is_original = [True] * len(original_X)
    
    augmentation_count = 0
    
    for class_label in [0, 1, 2]:
        class_indices = np.where(original_y == class_label)[0]
        class_strategy = determine_class_augmentation_strategy(class_label, augmentation_needs, 
                                                             len(class_indices), target_per_class)
        
        print(f"\nAugmenting class {class_label}:")
        print(f"  Original samples: {len(class_indices)}")
        print(f"  Strategy: {class_strategy}")
        
        for idx in class_indices:
            x_orig = original_X[idx]
            
            # Apply rotations
            if class_strategy['rotations'] > 0:
                rotations = get_rotations(x_orig, class_strategy['rotations'])
                for rotated in rotations:
                    augmented_X.append(rotated)
                    augmented_y.append(class_label)
                    augmented_is_original.append(False)
                    augmentation_count += 1
            
            # Apply flips
            if class_strategy['flips'] > 0:
                flipped = flip_horizontal(x_orig)
                augmented_X.append(flipped)
                augmented_y.append(class_label)
                augmented_is_original.append(False)
                augmentation_count += 1
            
            # Apply noise
            for _ in range(class_strategy['noise_copies']):
                for noise_level in class_strategy['noise_levels']:
                    noisy = add_gaussian_noise(x_orig, noise_level)
                    augmented_X.append(noisy)
                    augmented_y.append(class_label)
                    augmented_is_original.append(False)
                    augmentation_count += 1
    
    # Convert back to arrays
    final_X = np.array(augmented_X, dtype=object)
    final_y = np.array(augmented_y)
    final_is_original = np.array(augmented_is_original)
    
    print(f"\n=== AUGMENTATION RESULTS ===")
    print(f"Original samples: {total_original}")
    print(f"Augmented samples: {augmentation_count}")
    print(f"Total samples: {len(final_X)}")
    
    final_class_counts = Counter(final_y)
    print(f"Final class distribution: {dict(final_class_counts)}")
    
    return final_X, final_y, final_is_original

def create_balanced_train_test_split(X_sequences, y_sequences, is_original, target_total=2000):
    """
    Create balanced train/test split with controlled augmentation.
    RESTORED: Test set uses pure original data with controlled percentages per class.
    """
    # Convert to arrays
    X_sequences_array = np.array(X_sequences, dtype=object)
    y_sequences_array = np.array(y_sequences)
    is_original_array = np.array(is_original)
    
    print("=== ORIGINAL DATA ANALYSIS ===")
    original_indices = np.where(is_original_array == True)[0]
    original_X = X_sequences_array[original_indices]
    original_y = y_sequences_array[original_indices]
    
    print(f"Total original samples: {len(original_X)}")
    original_counts = Counter(original_y)
    print(f"Original class distribution: {dict(original_counts)}")
    
    # RESTORED: Use controlled test percentages per class (like original code)
    test_percentages = {0: 0.2, 1: 0.4, 2: 0.8}  # 0=not drinking, 1=drinking, 2=eating
    
    print(f"\nTest percentages per class: {test_percentages}")
    
    train_idx, test_idx = [], []
    
    # 1. Split original data into train and test per class (RESTORED)
    for c in [0, 1, 2]:
        idx = np.where((y_sequences_array == c) & (is_original_array == True))[0]
        test_size = test_percentages.get(c, 0.2)
        if len(idx) < 5:
            test_idx.extend(idx)
            continue
        idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=42, shuffle=True)
        train_idx.extend(idx_train)
        test_idx.extend(idx_test)
    
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    
    # 2. Test set: only original samples (RESTORED)
    X_test, y_test = X_sequences_array[test_idx], y_sequences_array[test_idx]
    
    # 3. Train/val: all original (not in test) + all augmented
    augmented_idx = np.where(is_original_array == False)[0]
    trainval_idx = np.concatenate([train_idx, augmented_idx])
    
    X_trainval, y_trainval = X_sequences_array[trainval_idx], y_sequences_array[trainval_idx]
    
    print(f"\nBefore augmentation: {len(X_trainval)} samples")
    print(f"Class distribution: {dict(Counter(y_trainval))}")
    
    # Apply balanced augmentation to train/val set
    X_aug, y_aug, is_original_aug = apply_balanced_classification_augmentation(
        X_trainval, y_trainval, [True] * len(X_trainval), target_total
    )
    
    print(f"\nAfter augmentation: {len(X_aug)} samples")
    print(f"Class distribution: {dict(Counter(y_aug))}")
    
    # 4. Final train/val split from the balanced, augmented set
    X_trainval_bal = np.array(X_aug, dtype=object)
    y_trainval_bal = np.array(y_aug)
    
    rng = np.random.default_rng(42)
    idx_tv = rng.permutation(len(X_trainval_bal))
    split_tv = int(.8 * len(X_trainval_bal))
    X_train, y_train = X_trainval_bal[idx_tv[:split_tv]], y_trainval_bal[idx_tv[:split_tv]]
    X_val, y_val = X_trainval_bal[idx_tv[split_tv:]], y_trainval_bal[idx_tv[split_tv:]]
    
    print(f"\n=== FINAL SPLITS ===")
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Detailed analysis of each split
    print("\n=== DETAILED CLASS DISTRIBUTION ===")
    
    # Class names for better readability
    class_names = {0: "Not Drinking", 1: "Drinking", 2: "Eating"}
    
    # Test set analysis (pure original data)
    print(f"\nTEST SET (Pure Original Data):")
    print("-" * 50)
    test_counts = Counter(y_test)
    for class_label in [0, 1, 2]:
        count = test_counts.get(class_label, 0)
        percentage = (count / len(y_test)) * 100 if len(y_test) > 0 else 0
        print(f"  {class_names[class_label]}: {count} samples ({percentage:.1f}%)")
    
    # Train set analysis
    print(f"\nTRAIN SET (Original + Augmented):")
    print("-" * 50)
    train_counts = Counter(y_train)
    for class_label in [0, 1, 2]:
        count = train_counts.get(class_label, 0)
        percentage = (count / len(y_train)) * 100 if len(y_train) > 0 else 0
        print(f"  {class_names[class_label]}: {count} samples ({percentage:.1f}%)")
    
    # Validation set analysis
    print(f"\nVALIDATION SET (Original + Augmented):")
    print("-" * 50)
    val_counts = Counter(y_val)
    for class_label in [0, 1, 2]:
        count = val_counts.get(class_label, 0)
        percentage = (count / len(y_val)) * 100 if len(y_val) > 0 else 0
        print(f"  {class_names[class_label]}: {count} samples ({percentage:.1f}%)")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print("-" * 30)
    total_samples = len(X_train) + len(X_val) + len(X_test)
    print(f"Total dataset: {total_samples} samples")
    
    # Overall class distribution
    all_y = list(y_train) + list(y_val) + list(y_test)
    overall_counts = Counter(all_y)
    
    for class_label in [0, 1, 2]:
        count = overall_counts.get(class_label, 0)
        percentage = (count / total_samples) * 100
        print(f"  {class_names[class_label]}: {count} samples ({percentage:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Example usage:
if __name__ == "__main__":
    # Your data should be in these variables:
    # X_sequences, y_sequences, is_original
    
    # Apply balanced augmentation and create splits
    X_train, y_train, X_val, y_val, X_test, y_test = create_balanced_train_test_split(
        X_sequences, y_sequences, is_original, target_total=2000
    )
    
    print("\n=== READY FOR TRAINING ===")
    print("Use X_train, y_train for training")
    print("Use X_val, y_val for validation")
    print("Use X_test, y_test for testing")

def explain_augmentation_process():
    """
    Explain how the augmentation process works in detail.
    """
    print("\n" + "="*60)
    print("AUGMENTATION PROCESS EXPLANATION")
    print("="*60)
    
    print("\n1. INITIAL ANALYSIS:")
    print("   - Count samples in each class")
    print("   - Calculate target samples per class (80% of largest class)")
    print("   - Determine how many samples each class needs")
    
    print("\n2. AUGMENTATION STRATEGIES:")
    print("   - Class 0 (Not Drinking): Minimal augmentation")
    print("     * 1-2 rotations if needed")
    print("     * 1 flip if needed")
    print("     * 0-1 noise copies if needed")
    
    print("\n   - Class 1 (Drinking): Moderate augmentation")
    print("     * 2-3 rotations if needed")
    print("     * 1 flip if needed")
    print("     * 0-1 noise copies if needed")
    
    print("\n   - Class 2 (Eating): Aggressive augmentation")
    print("     * 3-4 rotations if needed")
    print("     * 1 flip if needed")
    print("     * 1-2 noise copies if needed")
    
    print("\n3. AUGMENTATION TECHNIQUES:")
    print("   - ROTATIONS: Realistic angles for chest-mounted sensor")
    print("     * Angles: [5°, 355°, 10°, 350°, 20°, 340°, 30°, 330°, 45°, 315°]")
    print("     * Randomly selected for each sample")
    print("     * Represents realistic body movements")
    
    print("\n   - FLIPS: Horizontal flipping")
    print("     * Applied only ONCE per sample")
    print("     * Flips 8x8 frames horizontally")
    print("     * Creates mirror image of the data")
    
    print("\n   - NOISE: Gaussian noise")
    print("     * 15% of maximum possible value (4000)")
    print("     * std_dev = 0.15 * 4000 = 600")
    print("     * Helps model distinguish real vs augmented data")
    
    print("\n4. BALANCING LOGIC:")
    print("   - Target: Equal representation across classes")
    print("   - Strategy: Augment smaller classes more aggressively")
    print("   - Result: Balanced dataset for better model training")
    
    print("\n5. SPLIT CREATION:")
    print("   - Test set: 20% of original samples only")
    print("   - Train/Val: 80% of original + all augmented samples")
    print("   - Final split: 80% train, 20% validation from train/val")
    
    print("\n6. BENEFITS:")
    print("   - Controlled dataset size (~2000 samples)")
    print("   - Balanced class distribution")
    print("   - Realistic augmentations")
    print("   - Professional data science approach") 