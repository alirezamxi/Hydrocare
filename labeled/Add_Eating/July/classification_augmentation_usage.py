# ====================================================================
# USAGE EXAMPLE FOR BALANCED CLASSIFICATION AUGMENTATION (FIXED)
# ====================================================================
# This restores your original strategy: test set = pure original data

# Import the balanced augmentation functions
from balanced_classification_augmentation import (
    create_balanced_train_test_split
)

# Your current data (assuming you have these variables):
# X_sequences, y_sequences, is_original

print("=== ORIGINAL DATA ===")
print(f"Total samples: {len(X_sequences)}")
print(f"Class distribution: {Counter(y_sequences)}")

# Apply balanced augmentation and create splits (RESTORED STRATEGY)
X_train, y_train, X_val, y_val, X_test, y_test = create_balanced_train_test_split(
    X_sequences, y_sequences, is_original, target_total=2000
)

# Convert to lists if needed for your model
X_train = list(X_train)
X_val = list(X_val)
X_test = list(X_test)

print("\n=== READY FOR TRAINING ===")
print("Use X_train, y_train for training")
print("Use X_val, y_val for validation")
print("Use X_test, y_test for testing")

# ====================================================================
# DETAILED ANALYSIS FUNCTION
# ====================================================================
def analyze_splits_detailed(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Detailed analysis of class distribution in each split.
    """
    from collections import Counter
    
    print("\n" + "="*60)
    print("DETAILED CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Class names for better readability
    class_names = {0: "Not Drinking", 1: "Drinking", 2: "Eating"}
    
    # Analyze each split
    splits = {
        "TRAIN": (X_train, y_train),
        "VALIDATION": (X_val, y_val), 
        "TEST": (X_test, y_test)
    }
    
    for split_name, (X_split, y_split) in splits.items():
        print(f"\n{split_name} SET:")
        print("-" * 40)
        print(f"Total samples: {len(X_split)}")
        
        class_counts = Counter(y_split)
        total = len(y_split)
        
        for class_label in [0, 1, 2]:
            count = class_counts.get(class_label, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {class_names[class_label]}: {count} samples ({percentage:.1f}%)")
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print("-" * 40)
    total_samples = len(X_train) + len(X_val) + len(X_test)
    print(f"Total dataset: {total_samples} samples")
    
    # Overall class distribution
    all_y = list(y_train) + list(y_val) + list(y_test)
    overall_counts = Counter(all_y)
    
    for class_label in [0, 1, 2]:
        count = overall_counts.get(class_label, 0)
        percentage = (count / total_samples) * 100
        print(f"  {class_names[class_label]}: {count} samples ({percentage:.1f}%)")

# ====================================================================
# SIMPLE REPLACEMENT FOR YOUR CURRENT CODE
# ====================================================================
# Replace your current augmentation code with this simple approach:

"""
# OLD CODE (commented out):
# Set how many noise augmentations you want per class
# noise_augmentations = {0: 0, 1: 1, 2: 7}  # Example: more for eating

# Set rotation angles (unique, no repeats)
# rotation_angles = [5, -5, 10, -10, 20, -20, 30, -30, 45, -45]

# aug_X, aug_y = list(X_trainval), list(y_trainval)

# for c in [0, 1, 2]:
#     idx = np.where(y_trainval == c)[0]
#     for orig_idx in idx:
#         x_orig = X_trainval[orig_idx]
#         # 1. Flip (once)
#         x_flip = flip_horizontal(x_orig)
#         aug_X.append(x_flip)
#         aug_y.append(c)
#         # 2. Rotations (once per angle)
#         for angle in rotation_angles:
#             x_rot = rotate_sip(x_orig, angle)
#             aug_X.append(x_rot)
#             aug_y.append(c)
#         # 3. Noise (as many times as you want for this class)
#         for _ in range(noise_augmentations[c]):
#             x_noise = add_gaussian_noise(x_orig)
#             aug_X.append(x_noise)
#             aug_y.append(c)

# NEW CODE (simple replacement):
from balanced_classification_augmentation import create_balanced_train_test_split

# Apply balanced augmentation and create splits
X_train, y_train, X_val, y_val, X_test, y_test = create_balanced_train_test_split(
    X_sequences, y_sequences, is_original, target_total=2000
)

# Convert to lists if needed
X_train = list(X_train)
X_val = list(X_val)
X_test = list(X_test)
""" 