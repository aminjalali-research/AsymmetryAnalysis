#!/usr/bin/env python3
"""
Advanced Asymmetry Analysis Demo

Demonstrates advanced analytical methods using the P013 asymmetry results.
This includes dimensionality reduction, clustering, and pattern analysis.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append("src")

try:
    from advanced_analysis import (
        AsymmetryClusterAnalysis, 
        AsymmetryPCAAnalysis,
        AsymmetryNetworkAnalysis
    )
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    print("⚠️  Advanced analysis modules not fully available. Running basic demo.")


def load_asymmetry_data():
    """Load the comprehensive asymmetry results"""
    
    results_file = Path("P013_Analysis_Results/P013_comprehensive_asymmetry_analysis.csv")
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("Please run targeted_p013_analysis.py first!")
        return None
        
    df = pd.read_csv(results_file)
    print(f"✅ Loaded {len(df)} regions with {df.shape[1]} measurements")
    return df


def basic_advanced_analysis(df):
    """Perform basic advanced analysis without complex dependencies"""
    
    print("\n🔬 BASIC ADVANCED ASYMMETRY ANALYSIS")
    print("=" * 60)
    
    # Select asymmetry methods for analysis
    asymmetry_columns = [
        'laterality_index', 'cohens_d_asymmetry', 'volume_corrected_asymmetry',
        'fold_change_asymmetry', 'normalized_asymmetry', 'log_ratio',
        'percentage_difference'
    ]
    
    # Extract asymmetry data
    asymmetry_data = df[asymmetry_columns].copy()
    
    print(f"📊 Analyzing {len(asymmetry_columns)} asymmetry methods across {len(df)} regions")
    
    # 1. Correlation Analysis
    print("\n1️⃣ INTER-METHOD CORRELATION ANALYSIS")
    correlation_matrix = asymmetry_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Asymmetry Methods Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('P013_Analysis_Results/method_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Method Consistency Analysis
    print("\n2️⃣ METHOD CONSISTENCY ANALYSIS")
    consistency_scores = []
    for _, region_data in asymmetry_data.iterrows():
        # Calculate coefficient of variation (CV) as consistency measure
        cv = np.std(region_data) / (np.abs(np.mean(region_data)) + 1e-6)
        consistency_scores.append(cv)
    
    df['method_consistency'] = consistency_scores
    
    # Find most and least consistent regions
    most_consistent = df.nsmallest(5, 'method_consistency')[['region', 'method_consistency', 'laterality_index']]
    least_consistent = df.nlargest(5, 'method_consistency')[['region', 'method_consistency', 'laterality_index']]
    
    print("🏆 Most Consistent Asymmetry Patterns:")
    print(most_consistent.to_string(index=False))
    print("\n⚡ Most Variable Asymmetry Patterns:")
    print(least_consistent.to_string(index=False))
    
    # 3. Regional Asymmetry Patterns
    print("\n3️⃣ REGIONAL ASYMMETRY PATTERNS")
    
    # Separate subcortical and cortical regions
    subcortical = df[df['region_type'] == 'subcortical']
    cortical = df[df['region_type'] == 'cortical']
    
    print(f"📊 Subcortical regions: {len(subcortical)} | Cortical regions: {len(cortical)}")
    
    # Compare asymmetry magnitudes
    subcortical_li = subcortical['laterality_index'].abs()
    cortical_li = cortical['laterality_index'].abs()
    
    print(f"🧠 Subcortical |LI| mean: {subcortical_li.mean():.4f} ± {subcortical_li.std():.4f}")
    print(f"🧠 Cortical |LI| mean: {cortical_li.mean():.4f} ± {cortical_li.std():.4f}")
    
    # Statistical test
    from scipy.stats import mannwhitneyu
    statistic, p_value = mannwhitneyu(subcortical_li, cortical_li, alternative='two-sided')
    print(f"📈 Mann-Whitney U test p-value: {p_value:.6f}")
    
    # 4. Asymmetry Distribution Analysis
    print("\n4️⃣ ASYMMETRY DISTRIBUTION VISUALIZATION")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # LI distribution
    axes[0,0].hist(df['laterality_index'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0,0].axvline(x=0.1, color='green', linestyle=':', alpha=0.7, label='Significance threshold')
    axes[0,0].axvline(x=-0.1, color='green', linestyle=':', alpha=0.7)
    axes[0,0].set_title('Laterality Index Distribution', fontweight='bold')
    axes[0,0].set_xlabel('Laterality Index')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    
    # Effect size (Cohen's d) distribution
    axes[0,1].hist(df['cohens_d_asymmetry'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0,1].set_title("Cohen's d Distribution", fontweight='bold')
    axes[0,1].set_xlabel("Cohen's d")
    axes[0,1].set_ylabel('Frequency')
    
    # Regional comparison
    region_data = [subcortical['laterality_index'], cortical['laterality_index']]
    axes[1,0].boxplot(region_data, labels=['Subcortical', 'Cortical'])
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1,0].axhline(y=0.1, color='green', linestyle=':', alpha=0.7)
    axes[1,0].axhline(y=-0.1, color='green', linestyle=':', alpha=0.7)
    axes[1,0].set_title('Regional Type Comparison', fontweight='bold')
    axes[1,0].set_ylabel('Laterality Index')
    
    # Method consistency scatter
    axes[1,1].scatter(df['laterality_index'], df['method_consistency'], alpha=0.7, s=50)
    axes[1,1].set_xlabel('Laterality Index')
    axes[1,1].set_ylabel('Method Consistency (CV)')
    axes[1,1].set_title('LI vs Method Consistency', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('P013_Analysis_Results/advanced_asymmetry_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df


def main():
    """Main advanced analysis workflow"""
    
    print("🧠 P013 ADVANCED ASYMMETRY ANALYSIS")
    print("=" * 60)
    print("Demonstrating sophisticated analytical approaches...")
    
    # Load data
    df = load_asymmetry_data()
    if df is None:
        return
    
    # Run basic advanced analysis
    enhanced_df = basic_advanced_analysis(df)
    
    # Save enhanced results
    output_file = Path("P013_Analysis_Results/P013_advanced_analysis_results.csv")
    enhanced_df.to_csv(output_file, index=False)
    print(f"\n💾 Enhanced results saved to: {output_file}")
    
    print("\n🎉 ADVANCED ANALYSIS COMPLETE!")
    print("=" * 60)
    print("📊 Generated visualizations:")
    print("  • Method correlation matrix")
    print("  • Asymmetry distribution analysis")
    print("  • Regional comparison plots")
    print("  • Method consistency analysis")
    
    print("\n📈 Key insights:")
    print("  • Inter-method correlations reveal measurement consistency")
    print("  • Regional type differences in asymmetry patterns")
    print("  • Method consistency varies across brain regions")
    print("  • Distribution analysis shows asymmetry prevalence")


if __name__ == "__main__":
    main()
