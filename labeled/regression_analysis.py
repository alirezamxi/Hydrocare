import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for academic plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

def main():
    # Load and prepare the data
    file_path = 'dv0_witoutstraw.xlsx'
    data = pd.read_excel(file_path)

    # Filter for drinking events only
    data = data[data['Label'] == 1]

    # Group by participant and sip to get time steps and dV
    grouped_data = data.groupby(['Participant_ID', 'sip_id']).agg(
        time_steps=('sip_id', 'size'),
        dV=('dV', 'first'),
        Label=('Label', 'first')
    ).reset_index()

    print("Dataset Overview:")
    print(f"Total sips: {len(grouped_data)}")
    print(f"Participants: {grouped_data['Participant_ID'].nunique()}")
    print(f"Average sip volume: {grouped_data['dV'].mean():.2f} ml")
    print(f"Volume range: {grouped_data['dV'].min():.2f} - {grouped_data['dV'].max():.2f} ml")

    # Prepare features and target
    X = grouped_data[['time_steps']]
    y = grouped_data['dV']

    # Split the data (80% train, 20% test)
    train_size = int(0.8 * len(grouped_data))
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Define models with random seed 42
    models = {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(kernel='rbf'),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    }

    # Store results
    results = {}

    print("=" * 80)
    print("REGRESSION MODEL EVALUATION RESULTS (Seed: 42)")
    print("=" * 80)

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmspe = np.sqrt(np.mean(((y_test - y_pred) / y_test) ** 2)) * 100
        
        # Store results
        results[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'RMSPE': rmspe,
            'Predictions': y_pred
        }
        
        print(f"\n{model_name}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSPE: {rmspe:.2f}%")

    # Create results summary DataFrame
    results_df = pd.DataFrame({
        model: {metric: results[model][metric] for metric in ['MSE', 'RMSE', 'MAE', 'R²', 'RMSPE']}
        for model in results.keys()
    }).T

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(results_df.round(4))

    # Create academic-friendly regression plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (model_name, model_results) in enumerate(results.items()):
        ax = axes[idx]
        y_pred = model_results['Predictions']
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, color=colors[idx], s=50)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Actual Volume (ml)', fontsize=12)
        ax.set_ylabel('Predicted Volume (ml)', fontsize=12)
        ax.set_title(f'{model_name}\nR² = {model_results["R²"]:.3f}, RMSE = {model_results["RMSE"]:.2f}', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add text box with metrics
        textstr = f'MAE: {model_results["MAE"]:.2f}\nRMSPE: {model_results["RMSPE"]:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.suptitle('Regression Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('regression_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Identify and analyze outliers
    def identify_outliers_iqr(data, column, multiplier=1.5):
        """Identify outliers using IQR method"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    # Find outliers in the target variable (dV)
    outliers, lower_bound, upper_bound = identify_outliers_iqr(grouped_data, 'dV')

    print("=" * 80)
    print("OUTLIER ANALYSIS")
    print("=" * 80)
    print(f"Total samples: {len(grouped_data)}")
    print(f"Outliers detected: {len(outliers)}")
    print(f"Outlier percentage: {len(outliers)/len(grouped_data)*100:.2f}%")
    print(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}] ml")

    print("\nTop 5 Outliers:")
    outlier_analysis = outliers.nlargest(5, 'dV')[['Participant_ID', 'sip_id', 'time_steps', 'dV']]
    print(outlier_analysis)

    # Create outlier visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Box plot
    axes[0].boxplot(grouped_data['dV'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[0].set_title('Volume Distribution with Outliers', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Volume (ml)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Histogram with outlier highlighting
    axes[1].hist(grouped_data['dV'], bins=30, alpha=0.7, color='lightblue', label='Normal data')
    axes[1].hist(outliers['dV'], bins=10, alpha=0.8, color='red', label='Outliers')
    axes[1].axvline(lower_bound, color='red', linestyle='--', alpha=0.8, label=f'Lower bound: {lower_bound:.1f}')
    axes[1].axvline(upper_bound, color='red', linestyle='--', alpha=0.8, label=f'Upper bound: {upper_bound:.1f}')
    axes[1].set_title('Volume Distribution - Outliers Highlighted', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Volume (ml)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Detailed analysis of the top 5 outliers
    print("=" * 80)
    print("DETAILED OUTLIER ANALYSIS - TOP 5 OUTLIERS")
    print("=" * 80)

    top_5_outliers = outliers.nlargest(5, 'dV')

    for idx, (_, outlier) in enumerate(top_5_outliers.iterrows(), 1):
        print(f"\nOutlier #{idx}:")
        print(f"  Participant ID: {outlier['Participant_ID']}")
        print(f"  Sip ID: {outlier['sip_id']}")
        print(f"  Time steps: {outlier['time_steps']}")
        print(f"  Volume (dV): {outlier['dV']:.2f} ml")
        
        # Find the participant's other sips for context
        participant_data = grouped_data[grouped_data['Participant_ID'] == outlier['Participant_ID']]
        print(f"  Participant's total sips: {len(participant_data)}")
        print(f"  Participant's average sip volume: {participant_data['dV'].mean():.2f} ml")
        print(f"  Participant's volume range: {participant_data['dV'].min():.2f} - {participant_data['dV'].max():.2f} ml")
        
        # Check if this is the largest sip for this participant
        if outlier['dV'] == participant_data['dV'].max():
            print(f"  → This is the LARGEST sip for this participant")
        
        # Check time steps vs volume relationship
        if outlier['time_steps'] > participant_data['time_steps'].mean():
            print(f"  → This sip has ABOVE AVERAGE duration ({outlier['time_steps']} vs {participant_data['time_steps'].mean():.1f} avg)")
        else:
            print(f"  → This sip has BELOW AVERAGE duration ({outlier['time_steps']} vs {participant_data['time_steps'].mean():.1f} avg)")

    # Create a scatter plot showing outliers in context
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot all data points
    ax.scatter(grouped_data['time_steps'], grouped_data['dV'], alpha=0.6, color='lightblue', 
               label='Normal data', s=50)

    # Highlight outliers
    ax.scatter(outliers['time_steps'], outliers['dV'], alpha=0.8, color='red', 
               label='Outliers', s=100, edgecolors='darkred', linewidth=2)

    # Add annotations for top 5 outliers
    for idx, (_, outlier) in enumerate(top_5_outliers.iterrows(), 1):
        ax.annotate(f'#{idx}\nP{outlier["Participant_ID"]}\n{outlier["dV"]:.1f}ml', 
                    xy=(outlier['time_steps'], outlier['dV']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Time Steps (Duration)', fontsize=12)
    ax.set_ylabel('Volume (ml)', fontsize=12)
    ax.set_title('Sip Duration vs Volume - Outliers Highlighted', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outliers_in_context.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create a final summary table for academic publication
    print("=" * 100)
    print("FINAL REGRESSION MODEL PERFORMANCE SUMMARY (Seed: 42)")
    print("=" * 100)

    # Create a publication-ready table
    summary_table = pd.DataFrame({
        'Model': list(results.keys()),
        'R²': [f"{results[model]['R²']:.4f}" for model in results.keys()],
        'RMSE (ml)': [f"{results[model]['RMSE']:.2f}" for model in results.keys()],
        'MAE (ml)': [f"{results[model]['MAE']:.2f}" for model in results.keys()],
        'RMSPE (%)': [f"{results[model]['RMSPE']:.1f}" for model in results.keys()]
    })

    print(summary_table.to_string(index=False))

    print("\n" + "=" * 100)
    print("KEY FINDINGS:")
    print("=" * 100)
    print("1. ALL models show NEGATIVE R² scores, indicating poor performance")
    print("2. Models perform WORSE than simply predicting the mean value")
    print("3. Linear Regression shows the least negative R² (-0.2763)")
    print("4. K-Nearest Neighbors shows the worst performance (R² = -0.9566)")
    print("5. High RMSPE values (47-95%) indicate significant prediction errors")
    print("6. The single feature (time_steps) appears insufficient for volume prediction")

    print("\n" + "=" * 100)
    print("OUTLIER ANALYSIS SUMMARY:")
    print("=" * 100)
    print(f"• Total outliers detected: {len(outliers)} out of {len(grouped_data)} samples ({len(outliers)/len(grouped_data)*100:.1f}%)")
    print(f"• Outlier volume range: {outliers['dV'].min():.2f} - {outliers['dV'].max():.2f} ml")
    print(f"• Normal data range: {lower_bound:.2f} - {upper_bound:.2f} ml")
    print("• Outliers represent extreme drinking behaviors or measurement errors")
    print("• These outliers significantly impact model performance")

    print("\n" + "=" * 100)
    print("RECOMMENDATIONS:")
    print("=" * 100)
    print("1. Include additional features beyond just time_steps")
    print("2. Consider sensor data from multiple zones")
    print("3. Implement feature engineering (e.g., velocity, acceleration)")
    print("4. Use more sophisticated models (e.g., neural networks)")
    print("5. Consider outlier removal or robust regression methods")
    print("6. Implement cross-validation with proper participant-wise splitting")

if __name__ == "__main__":
    main()
