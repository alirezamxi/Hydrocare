# 🎯 Middle Performer Model - Paper Ready

## 📊 Selection Summary

**Selected Model**: Seed 42  
**Reason**: Middle-performing model among 10 random seeds  
**F1 Score**: 0.9207 (5th best out of 10 seeds)

## 🏆 Performance Ranking (F1 Scores)

| Rank | Seed | F1 Score | Status |
|------|------|----------|---------|
| 🥇 1st | 101 | 0.9489 | Best |
| 🥈 2nd | 606 | 0.9394 | 2nd Best |
| 🥉 3rd | 303 | 0.9348 | 3rd Best |
| 4th | 123 | 0.9344 | 4th Best |
| **5th** | **42** | **0.9207** | **🔄 MIDDLE (SELECTED)** |
| 6th | 404 | 0.9300 | 6th Best |
| 7th | 789 | 0.9166 | 7th Best |
| 8th | 202 | 0.9090 | 8th Best |
| 9th | 505 | 0.9017 | 9th Best |
| 10th | 456 | 0.8968 | Worst |

## 📈 Model Performance Details

### Overall Metrics
- **F1 Score**: 0.9207
- **Accuracy**: 0.9207
- **Precision**: 0.9230
- **Recall**: 0.9207
- **Total Samples**: 429

### Per-Class Performance
| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Not_Drinking | 0.9237 | 0.9528 | 0.9380 |
| Drinking | 0.8812 | 0.9400 | 0.9097 |
| Eating | 0.9638 | 0.8750 | 0.9172 |

### Confusion Matrix
```
                Predicted
                Not_Drinking  Drinking  Eating
True
Not_Drinking       121          5        1
Drinking             5        141        4
Eating               5         14      133
```

## 📁 Files Included

### 🎓 **Academic Confusion Matrices (Publication Ready)**
1. **`final_academic_confusion_matrix.png/.pdf`** - **RECOMMENDED** - Clean academic style with large numbers
2. **`ultra_large_academic_confusion_matrix.png/.pdf`** - Ultra-large numbers that fill the cells
3. **`academic_confusion_matrix.png/.pdf`** - Standard academic version
4. **`academic_confusion_matrix_large.png/.pdf`** - Large academic version

### 📊 **Model and Data Files**
5. **`middle_performer_model.keras`** - The trained TCN model
6. **`middle_performer_results.json`** - Complete performance metrics
7. **`middle_performer_predictions.xlsx`** - All predictions and true labels
8. **`performance_comparison_all_seeds.png`** - Comprehensive performance comparison across all seeds

## 🎯 Why This Model Was Selected

This model represents the **middle performance** among all 10 random seeds, which is ideal for academic papers because:

1. **Balanced Performance**: Not overly optimistic (like the best performer) or pessimistic (like the worst)
2. **Representative**: Shows typical performance you can expect from the model architecture
3. **Robust**: Middle performance suggests stability across different random initializations
4. **Academic Integrity**: Avoids cherry-picking the best result, which could be misleading

## 📊 Statistical Context

- **Mean F1 Score**: 0.9232
- **Standard Deviation**: 0.0167
- **Selected Model**: 0.9207 (within 1 standard deviation of mean)
- **Range**: 0.8968 - 0.9489

## 🔬 For Your Paper

### Citation Note
When using this model in your paper, you can state:
> "We selected the middle-performing model (Seed 42, F1 = 0.9207) from 10 random initializations to ensure representative performance reporting."

### Files to Include
- **RECOMMENDED**: Use `final_academic_confusion_matrix.png` or `ultra_large_academic_confusion_matrix.png` for the main confusion matrix (choose based on your preference for number size)
- Use `performance_comparison_all_seeds.png` to show the selection process
- Reference the metrics from `middle_performer_results.json`
- PDF versions are available for LaTeX documents

### Model Description
- **Architecture**: Temporal Convolutional Network (TCN)
- **Classes**: 3-class classification (Not_Drinking, Drinking, Eating)
- **Dataset Size**: 429 samples
- **Performance**: Balanced across all classes with F1 score of 0.9207

## 🚀 Usage

To load and use this model in your code:

```python
from tensorflow import keras

# Load the model
model = keras.models.load_model('middle_performer_model.keras')

# Make predictions
predictions = model.predict(your_data)
```

---

**Generated on**: August 20, 2025  
**Analysis Script**: `analyze_and_save_middle_performer.py`  
**Total Seeds Analyzed**: 10  
**Selection Criteria**: Middle F1 score performance 