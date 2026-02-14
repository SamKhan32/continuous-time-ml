import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_2d

# Load the processed data
processed_dir = "data/processed/"
pfl_table = pd.read_csv(processed_dir + "PFL_preprocessed.csv")

# Also process OSD if you want to compare
# osd_table = nc_convert(osd_data_og, processed_dir + "OSD_preprocessed.csv")
# osd_table = pd.read_csv(processed_dir + "OSD_preprocessed.csv")


def plot_observation_heatmap(df, title="Observation Density", figsize=(12, 8), bins=100):
    """
    Create a heatmap showing spatial distribution of observations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'lat' and 'lon' columns
    title : str
        Plot title
    figsize : tuple
        Figure size
    bins : int or tuple
        Number of bins for the heatmap (can be single int or (x_bins, y_bins))
    """
    # Remove NaN values
    df_clean = df.dropna(subset=['lat', 'lon'])
    
    # Get lat/lon ranges from your search criteria
    lon_range = [-97.5480, -4.7344]
    lat_range = [-14.0801, 54.1231]
    
    # Create 2D histogram
    counts, lon_edges, lat_edges, _ = binned_statistic_2d(
        df_clean['lon'], 
        df_clean['lat'], 
        None, 
        statistic='count',
        bins=bins,
        range=[lon_range, lat_range]
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(
        counts.T, 
        origin='lower',
        extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
        aspect='auto',
        cmap='hot',
        interpolation='nearest'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Number of Observations')
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics text
    stats_text = f"Total observations: {len(df_clean):,}\n"
    stats_text += f"Unique casts: {df_clean['castIndex'].nunique():,}"
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    return fig, ax


def plot_cast_heatmap(df, title="Cast Density", figsize=(12, 8), bins=100):
    """
    Create a heatmap showing spatial distribution of unique casts (not individual observations).
    """
    # Get one row per cast
    df_casts = df.groupby('castIndex').first().reset_index()
    df_casts = df_casts.dropna(subset=['lat', 'lon'])
    
    # Get lat/lon ranges
    lon_range = [-97.5480, -4.7344]
    lat_range = [-14.0801, 54.1231]
    
    # Create 2D histogram
    counts, lon_edges, lat_edges, _ = binned_statistic_2d(
        df_casts['lon'], 
        df_casts['lat'], 
        None, 
        statistic='count',
        bins=bins,
        range=[lon_range, lat_range]
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(
        counts.T, 
        origin='lower',
        extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
        aspect='auto',
        cmap='YlOrRd',
        interpolation='nearest'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Number of Casts')
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics
    stats_text = f"Total casts: {len(df_casts):,}"
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    return fig, ax


# Create visualizations for PFL
print("Creating PFL observation heatmap...")
fig1, ax1 = plot_observation_heatmap(pfl_table, title="PFL: Observation Density", bins=150)
plt.savefig(processed_dir + "PFL_observation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

print("Creating PFL cast heatmap...")
fig2, ax2 = plot_cast_heatmap(pfl_table, title="PFL: Cast Density", bins=150)
plt.savefig(processed_dir + "PFL_cast_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()


# If you also have OSD data, create comparison plots
# print("Creating OSD observation heatmap...")
# fig3, ax3 = plot_observation_heatmap(osd_table, title="OSD: Observation Density", bins=150)
# plt.savefig(processed_dir + "OSD_observation_heatmap.png", dpi=300, bbox_inches='tight')
# plt.show()

# print("Creating OSD cast heatmap...")
# fig4, ax4 = plot_cast_heatmap(osd_table, title="OSD: Cast Density", bins=150)
# plt.savefig(processed_dir + "OSD_cast_heatmap.png", dpi=300, bbox_inches='tight')
# plt.show()


# Create side-by-side comparison if you have both datasets
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
# # ... add comparison plots


# Print summary statistics
print("\n=== PFL Summary Statistics ===")
print(f"Total observations: {len(pfl_table):,}")
print(f"Total casts: {pfl_table['castIndex'].nunique():,}")
print(f"Lat range: [{pfl_table['lat'].min():.2f}, {pfl_table['lat'].max():.2f}]")
print(f"Lon range: [{pfl_table['lon'].min():.2f}, {pfl_table['lon'].max():.2f}]")
print(f"\nObservations per cast (avg): {len(pfl_table) / pfl_table['castIndex'].nunique():.1f}")