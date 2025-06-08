"""
Create Key Visualizations for Spotify Data
Run this to generate all exploration plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

def load_data():
    """Load the processed data"""
    # Try processed data first
    processed_path = Path("data/processed/spotify_clean.csv")
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        print(f"✅ Loaded processed data: {len(df):,} songs")
    else:
        # Fallback to raw data
        raw_path = Path("data/raw/Spotify_Data.csv")
        df = pd.read_csv(raw_path)
        print(f"✅ Loaded raw data: {len(df):,} songs")
    
    return df

def create_target_distribution_plot(df):
    """Plot 1: Target Distribution"""
    plt.figure(figsize=(10, 6))
    
    # Main plot
    plt.subplot(1, 2, 1)
    target_counts = df['target'].value_counts()
    plt.pie(target_counts.values, labels=['Non-Hit', 'Hit'], autopct='%1.1f%%',
            colors=['lightcoral', 'lightgreen'], startangle=90)
    plt.title('Song Success Distribution', fontsize=14, fontweight='bold')
    
    # Bar plot
    plt.subplot(1, 2, 2)
    target_counts.plot(kind='bar', color=['lightcoral', 'lightgreen'])
    plt.title('Hit vs Non-Hit Songs', fontsize=14, fontweight='bold')
    plt.xlabel('Target (0=Non-Hit, 1=Hit)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('results/figures/01_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print insights
    hit_rate = df['target'].mean()
    print(f"📈 INSIGHT: {hit_rate:.1%} of songs are hits")

def create_audio_features_comparison(df):
    """Plot 2: Audio Features by Success"""
    audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                     'liveness', 'speechiness', 'valence']
    
    # Filter to available features
    available_features = [f for f in audio_features if f in df.columns]
    
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(available_features, 1):
        plt.subplot(3, 3, i)
        
        # Box plot comparing hits vs non-hits
        df.boxplot(column=feature, by='target', ax=plt.gca())
        plt.title(f'{feature.title()}')
        plt.xlabel('Target (0=Non-Hit, 1=Hit)')
        plt.ylabel(feature)
        plt.suptitle('')  # Remove auto title
    
    plt.suptitle('Audio Features: Hits vs Non-Hits', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/02_audio_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_heatmap(df):
    """Plot 3: Feature Correlation Heatmap"""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find strong correlations with target
    if 'target' in numeric_cols:
        target_corr = corr_matrix['target'].abs().sort_values(ascending=False)
        print("\n🔍 STRONGEST CORRELATIONS WITH SUCCESS:")
        for feature, corr in target_corr.items():
            if feature != 'target' and abs(corr) > 0.1:
                direction = "positive" if corr > 0 else "negative"
                print(f"   {feature}: {corr:.3f} ({direction})")

def create_success_rate_by_feature(df):
    """Plot 4: Success Rate by Feature Quartiles"""
    audio_features = ['danceability', 'energy', 'valence', 'acousticness']
    available_features = [f for f in audio_features if f in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(available_features[:4]):
        ax = axes[i]
        
        # Create quartiles
        df[f'{feature}_quartile'] = pd.qcut(df[feature], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Calculate success rate by quartile
        success_rates = df.groupby(f'{feature}_quartile')['target'].mean()
        
        # Plot
        success_rates.plot(kind='bar', ax=ax, color='skyblue', alpha=0.7)
        ax.set_title(f'Success Rate by {feature.title()} Quartile', fontweight='bold')
        ax.set_ylabel('Hit Rate')
        ax.set_xlabel(f'{feature.title()} Quartile')
        ax.tick_params(axis='x', rotation=0)
        
        # Add value labels on bars
        for j, v in enumerate(success_rates.values):
            ax.text(j, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
    
    plt.suptitle('How Audio Features Affect Success Rate', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/04_success_by_features.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_distributions(df):
    """Plot 5: Feature Distribution Overlays"""
    audio_features = ['danceability', 'energy', 'valence', 'tempo']
    available_features = [f for f in audio_features if f in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(available_features[:4]):
        ax = axes[i]
        
        # Plot distributions for hits and non-hits
        hits = df[df['target'] == 1][feature]
        non_hits = df[df['target'] == 0][feature]
        
        ax.hist(non_hits, bins=30, alpha=0.7, label='Non-Hits', color='lightcoral', density=True)
        ax.hist(hits, bins=30, alpha=0.7, label='Hits', color='lightgreen', density=True)
        
        ax.set_title(f'{feature.title()} Distribution', fontweight='bold')
        ax.set_xlabel(feature.title())
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.suptitle('Feature Distributions: Hits vs Non-Hits', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/05_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_tempo_vs_energy_scatter(df):
    """Plot 6: Tempo vs Energy Scatter Plot"""
    if 'tempo' not in df.columns or 'energy' not in df.columns:
        print("⚠️ Tempo or Energy not available for scatter plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    hits = df[df['target'] == 1]
    non_hits = df[df['target'] == 0]
    
    plt.scatter(non_hits['tempo'], non_hits['energy'], alpha=0.6, 
               c='lightcoral', label='Non-Hits', s=30)
    plt.scatter(hits['tempo'], hits['energy'], alpha=0.8, 
               c='lightgreen', label='Hits', s=30)
    
    plt.xlabel('Tempo (BPM)', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title('Tempo vs Energy: Hits vs Non-Hits', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/06_tempo_energy_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_insights_summary(df):
    """Generate key insights from the data"""
    print("\n" + "="*50)
    print("🧠 KEY INSIGHTS FROM DATA EXPLORATION")
    print("="*50)
    
    # Basic stats
    hit_rate = df['target'].mean()
    print(f"📊 Overall hit rate: {hit_rate:.1%}")
    
    # Feature insights
    audio_features = ['danceability', 'energy', 'valence', 'acousticness']
    available_features = [f for f in audio_features if f in df.columns]
    
    print("\n🎵 AUDIO FEATURE INSIGHTS:")
    for feature in available_features:
        hit_mean = df[df['target'] == 1][feature].mean()
        non_hit_mean = df[df['target'] == 0][feature].mean()
        diff = hit_mean - non_hit_mean
        
        direction = "higher" if diff > 0 else "lower"
        print(f"   • Hit songs have {direction} {feature}: {hit_mean:.3f} vs {non_hit_mean:.3f}")
    
    # Correlation insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'target' in numeric_cols:
        corr_with_target = df[numeric_cols].corr()['target'].abs().sort_values(ascending=False)
        print(f"\n🔍 MOST PREDICTIVE FEATURES:")
        for feature, corr in corr_with_target.items():
            if feature != 'target' and abs(corr) > 0.05:
                print(f"   • {feature}: {corr:.3f} correlation")
    
    print("\n💡 BUSINESS INSIGHTS:")
    print("   • Songs with specific audio characteristics are more likely to be hits")
    print("   • We can build ML models to predict hit probability")
    print("   • A/B testing can help optimize song recommendations")

def main():
    """Run all visualizations"""
    print("🎨 CREATING SPOTIFY DATA VISUALIZATIONS")
    print("="*50)
    
    # Create results directory
    results_dir = Path("results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    print("\n1 Creating target distribution plot...")
    create_target_distribution_plot(df)
    
    print("\n2️ Creating audio features comparison...")
    create_audio_features_comparison(df)
    
    print("\n3️ Creating correlation heatmap...")
    create_correlation_heatmap(df)
    
    print("\n4️ Creating success rate by features...")
    create_success_rate_by_feature(df)
    
    print("\n5️ Creating feature distributions...")
    create_feature_distributions(df)
    
    print("\n6️ Creating tempo vs energy scatter...")
    create_tempo_vs_energy_scatter(df)
    
    print("\n7️ Generating insights summary...")
    generate_insights_summary(df)
    
    print(f"\n🎉 ALL VISUALIZATIONS COMPLETE!")
    print(f" Saved to: {results_dir}")
    print("\n Ready for feature engineering and ML modeling!")

if __name__ == "__main__":
    main()