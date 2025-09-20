#!/usr/bin/env python3
"""
TCN Regression Analysis - Using Existing Results
This script analyzes the existing TCN neural network results and creates academic-friendly plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set academic plotting style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 300

def load_tcn_results():
    """Load all existing TCN results"""
    results_dir = Path('Add_Eating/July/multi_seed_results')
    all_metrics = []
    
    print('🔍 Loading existing TCN results...')
    for seed_dir in results_dir.glob('seed_*'):
        seed = seed_dir.name.split('_')[1]
        metrics_file = seed_dir / 'metrics.json'
        predictions_file = seed_dir / 'predictions.xlsx'
        
        if metrics_file.exists() and predictions_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            # Load predictions for outlier analysis
            try:
                pred_df = pd.read_excel(predictions_file)
                y_test = pred_df['Actual_dV'].values
                y_pred = pred_df['Predicted_dV'].values
            except:
                y_test = None
                y_pred = None
            
            all_metrics.append({
                'seed': seed,
                'r2': metrics['r2'],
                'rmse': metrics['rmse'],
                'rmspe': metrics['rmspe'],
                'test_samples': metrics['test_samples'],
                'y_test': y_test,
                'y_pred': y_pred,
                'plots_dir': seed_dir / 'plots'
            })
            print(f'✅ Loaded Seed {seed}: R² = {metrics["r2"]:.4f}')
    
    return all_metrics

def create_ranking_table(all_metrics):
    """Create and display ranking table"""
    # Sort by R² score
    all_metrics.sort(key=lambda x: x['r2'], reverse=True)
    
    print(f'\n{"="*80}')
    print('🏆 TCN MODEL PERFORMANCE RANKING (All Seeds)')
    print(f'{"="*80}')
    print(f'{"Rank":<5} {"Seed":<8} {"R² Score":<10} {"RMSE":<10} {"RMSPE (%)":<12} {"Test Samples":<15}')
    print('-' * 80)
    
    for rank, metrics in enumerate(all_metrics, 1):
        print(f'{rank:<5} {metrics["seed"]:<8} {metrics["r2"]:<10.4f} {metrics["rmse"]:<10.2f} {metrics["rmspe"]:<12.1f} {metrics["test_samples"]:<15}')
    
    return all_metrics

def create_academic_plot(y_test, y_pred, r2, rmse, rmspe, save_path, seed_label=None):
    """Create academic-friendly plot with outlier annotation"""
    plt.figure(figsize=(8, 6), dpi=600)
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 26,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })
    
    # Main scatter plot (navy blue for academic style)
    plt.scatter(y_test, y_pred, alpha=0.85, color='#315c89', s=90, edgecolor='k', linewidth=1.0)
    
    # Perfect prediction line
    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    plt.plot(lims, lims, "--", color='#CC0000', linewidth=3, label='Perfect Prediction')
    plt.xlim(lims)
    plt.ylim(lims)
    
    # Labels and formatting
    plt.xlabel("Actual Drinking Volume (mL)", fontweight='bold', fontsize=24)
    plt.ylabel("Predicted Drinking Volume (mL)", fontweight='bold', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=False, fontsize=18, loc='upper left')
    
    # Find and annotate top 5 outliers
    abs_errors = np.abs(y_test - y_pred)
    sorted_indices = np.argsort(abs_errors)[::-1]
    seen_points = set()
    outlier_points = []
    
    for idx in sorted_indices:
        point = (float(y_test[idx]), float(y_pred[idx]))
        if point not in seen_points:
            outlier_points.append((idx, y_test[idx], y_pred[idx], abs_errors[idx]))
            seen_points.add(point)
        if len(outlier_points) == 5:
            break
    
    # Highlight outliers in red
    for idx, actual, pred, err in outlier_points:
        plt.scatter([actual], [pred], color='crimson', s=160, edgecolor='k', linewidth=2, zorder=5)
        plt.annotate(
            f"{int(actual)}→{int(pred)}",
            (actual, pred),
            textcoords="offset points",
            xytext=(10, -10),
            ha='left',
            fontsize=16,
            fontweight='bold',
            color='crimson',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="crimson", lw=1, alpha=0.7)
        )
    
    plt.tight_layout()
    
    # Save plot
    if seed_label is not None:
        pdf_path = save_path.parent / f"Final_regression_plot_seed_{seed_label}.pdf"
        png_path = save_path.parent / f"Final_regression_plot_seed_{seed_label}.png"
    else:
        pdf_path = save_path.parent / "Final_regression_plot.pdf"
        png_path = save_path.parent / "Final_regression_plot.png"
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print outlier information
    print(f"\nTop 5 Outliers (by absolute error) for Seed {seed_label}:")
    for i, (idx, actual, pred, err) in enumerate(outlier_points, 1):
        print(f"  Outlier {i}: Index {idx}, Actual = {actual:.2f}, Predicted = {pred:.2f}, Abs Error = {err:.2f}")
    
    return outlier_points

def create_comparison_plots(all_metrics):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² Score comparison
    ax1 = axes[0, 0]
    seeds = [m['seed'] for m in all_metrics]
    r2_values = [m['r2'] for m in all_metrics]
    
    bars = ax1.bar(range(len(seeds)), r2_values, color='skyblue', alpha=0.7, edgecolor='black', linewidth=1)
    bars[0].set_color('gold')  # Best
    bars[len(bars)//2].set_color('green')  # Middle
    bars[-1].set_color('red')  # Worst
    
    ax1.set_xlabel('Seed', fontweight='bold')
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('R² Scores Comparison Across Seeds', fontweight='bold')
    ax1.set_xticks(range(len(seeds)))
    ax1.set_xticklabels(seeds, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, r2_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE comparison
    ax2 = axes[0, 1]
    rmse_values = [m['rmse'] for m in all_metrics]
    bars2 = ax2.bar(range(len(seeds)), rmse_values, color='lightcoral', alpha=0.7, edgecolor='black', linewidth=1)
    bars2[0].set_color('gold')  # Best R²
    bars2[len(bars2)//2].set_color('green')  # Middle R²
    bars2[-1].set_color('red')  # Worst R²
    
    ax2.set_xlabel('Seed', fontweight='bold')
    ax2.set_ylabel('RMSE (mL)', fontweight='bold')
    ax2.set_title('RMSE Comparison Across Seeds', fontweight='bold')
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels(seeds, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # RMSPE comparison
    ax3 = axes[1, 0]
    rmspe_values = [m['rmspe'] if not np.isnan(m['rmspe']) else 0 for m in all_metrics]
    bars3 = ax3.bar(range(len(seeds)), rmspe_values, color='lightgreen', alpha=0.7, edgecolor='black', linewidth=1)
    bars3[0].set_color('gold')  # Best R²
    bars3[len(bars3)//2].set_color('green')  # Middle R²
    bars3[-1].set_color('red')  # Worst R²
    
    ax3.set_xlabel('Seed', fontweight='bold')
    ax3.set_ylabel('RMSPE (%)', fontweight='bold')
    ax3.set_title('RMSPE Comparison Across Seeds', fontweight='bold')
    ax3.set_xticks(range(len(seeds)))
    ax3.set_xticklabels(seeds, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Performance summary
    ax4 = axes[1, 1]
    best = all_metrics[0]
    worst = all_metrics[-1]
    middle_idx = len(all_metrics) // 2
    middle = all_metrics[middle_idx]
    
    summary_text = f"""Performance Summary

Total Seeds: {len(all_metrics)}
Best R²: {best['r2']:.4f} (Seed {best['seed']})
Worst R²: {worst['r2']:.4f} (Seed {worst['seed']})
Mean R²: {np.mean(r2_values):.4f}

Top 3 Seeds:
1. Seed {best['seed']}: R² = {best['r2']:.4f}
2. Seed {all_metrics[1]['seed']}: R² = {all_metrics[1]['r2']:.4f}
3. Seed {all_metrics[2]['seed']}: R² = {all_metrics[2]['r2']:.4f}

Model: TCN Neural Network
Data: Augmented (3.15x ratio)"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightblue", alpha=0.8))
    ax4.axis('off')
    ax4.set_title('Final Performance Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('TCN_performance_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig('TCN_performance_comparison.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()

def main():
    """Main analysis function"""
    print("="*80)
    print("TCN REGRESSION ANALYSIS - USING EXISTING RESULTS")
    print("="*80)
    print("Note: This analysis uses the existing TCN neural network results")
    print("with data augmentation, not traditional regression models.")
    print("="*80)
    
    # Load results
    all_metrics = load_tcn_results()
    
    if not all_metrics:
        print("❌ No TCN results found!")
        return
    
    # Create ranking
    all_metrics = create_ranking_table(all_metrics)
    
    # Show key statistics
    r2_scores = [m['r2'] for m in all_metrics]
    rmse_scores = [m['rmse'] for m in all_metrics]
    rmspe_scores = [m['rmspe'] for m in all_metrics if not np.isnan(m['rmspe'])]
    
    print(f'\n📊 STATISTICAL SUMMARY:')
    print(f'Best R²: {max(r2_scores):.4f} (Seed {all_metrics[0]["seed"]})')
    print(f'Worst R²: {min(r2_scores):.4f} (Seed {all_metrics[-1]["seed"]})')
    print(f'Mean R²: {np.mean(r2_scores):.4f}')
    print(f'Std R²: {np.std(r2_scores):.4f}')
    
    print(f'\nRMSE Scores:')
    print(f'  Mean: {np.mean(rmse_scores):.2f} mL')
    print(f'  Std:  {np.std(rmse_scores):.2f} mL')
    print(f'  Min:  {np.min(rmse_scores):.2f} mL')
    print(f'  Max:  {np.max(rmse_scores):.2f} mL')
    
    if rmspe_scores:
        print(f'\nRMSPE Scores:')
        print(f'  Mean: {np.mean(rmspe_scores):.1f}%')
        print(f'  Std:  {np.std(rmspe_scores):.1f}%')
        print(f'  Min:  {np.min(rmspe_scores):.1f}%')
        print(f'  Max:  {np.max(rmspe_scores):.1f}%')
    
    # Create comparison plots
    create_comparison_plots(all_metrics)
    
    # Create academic plot for middle performer
    middle_idx = len(all_metrics) // 2
    middle = all_metrics[middle_idx]
    
    print(f'\n{"="*80}')
    print(f"📈 CREATING ACADEMIC PLOT FOR MIDDLE PERFORMER (Seed {middle['seed']})")
    print(f"{'='*80}")
    
    if middle['y_test'] is not None and middle['y_pred'] is not None:
        save_path = middle['plots_dir'] / "Final_regression_plot"
        outlier_points = create_academic_plot(
            middle['y_test'], middle['y_pred'], 
            middle['r2'], middle['rmse'], middle['rmspe'], 
            save_path, middle['seed']
        )
        
        print(f"\n✅ Academic plot saved as: {save_path.with_suffix('.pdf')}")
    else:
        print("❌ Could not load prediction data for middle performer")
    
    # Save ranking to Excel
    ranking_df = pd.DataFrame(all_metrics)
    ranking_df = ranking_df.sort_values('r2', ascending=False)
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
    ranking_df = ranking_df[['Rank', 'seed', 'r2', 'rmse', 'rmspe', 'test_samples']]
    ranking_df.columns = ['Rank', 'Seed', 'R² Score', 'RMSE', 'RMSPE (%)', 'Test Samples']
    ranking_df.to_excel('TCN_seed_ranking_summary.xlsx', index=False)
    
    print(f"\n{'='*80}")
    print("🎉 ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    print("📁 Files created:")
    print("  - TCN_performance_comparison.pdf/png")
    print("  - TCN_seed_ranking_summary.xlsx")
    print(f"  - Academic plot for Seed {middle['seed']}")
    print(f"🏆 Best performing seed: {all_metrics[0]['seed']} (R² = {all_metrics[0]['r2']:.4f})")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
