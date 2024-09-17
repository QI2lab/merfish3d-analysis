import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
import matplotlib.style as mplstyle
mplstyle.use(['dark_background', 'ggplot', 'fast'])

# modify these two lines
data_path = Path('/mnt/data/bartelle/20240425_ECL_48CryoB_1_PL025') # should be top level path for dataset
data_to_use = 'sum_corrected_pixels' # options: 'sum_corrected_pixels' or 'sum_decon_pixels'

# 'sum_corrected_pixels' uses spot intensities from the camera after offset and gain correction
# 'sum_decon_pixels' uses spot intensities after deconvolution
# 'sum_corrected_pixels' is only available on datasets generated with v0.1.4 or later.

# -------------------------------------------------------------------------
# don't modify below here
ufish_path = data_path / 'processed_v2' / 'ufish_localizations'
tile_ids = sorted([entry.name for entry in ufish_path.iterdir() if entry.is_dir()])

first_tile_path = ufish_path / tile_ids[0]
file_info = []
for bit_file in sorted(first_tile_path.glob('bit*.parquet')):
    full_file_name = bit_file.name
    part_before_dot = bit_file.stem
    file_info.append((full_file_name, part_before_dot))

for bit_file_path, bit_id in file_info:
    temp_df = []
    for tile_id in tile_ids:
        temp_df.append(pd.read_parquet(ufish_path / tile_id / Path(bit_file_path)))
    df = pd.concat(temp_df, ignore_index=True)

    # Generate thresholds
    min_intensity = df[data_to_use].min()
    max_intensity = df[data_to_use].max()
    thresholds = np.linspace(min_intensity, max_intensity, 10000)

    # Calculate number of features retained for each threshold
    feature_counts = [len(df[(df[data_to_use] > threshold) & (df[data_to_use] < max_intensity)]) for threshold in thresholds]

    # Pre-calculate the histogram
    hist_bins = np.linspace(min_intensity, max_intensity, 100)
    n, bins = np.histogram(df[data_to_use], bins=hist_bins)
    hist_max_value = bins[np.argmax(n)]
    
    # Set initial thresholds
    init_lower_threshold = 1.1 * min_intensity
    init_upper_threshold = 0.5 * max_intensity

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, wspace=0.3)

    # First plot: Feature Retention vs. Intensity Threshold
    l, = ax1.plot(thresholds, feature_counts, lw=2)
    ax1.set_xlabel('Intensity Threshold')
    ax1.set_ylabel('Number of Features Retained')
    ax1.set_title('Feature Retention vs. Intensity Threshold; ' + str(bit_id))

    # Vertical lines and text on the first plot
    vline_lower = ax1.axvline(x=init_lower_threshold, color='r', linestyle='--')
    vline_upper = ax1.axvline(x=init_upper_threshold, color='b', linestyle='--')
    text = ax1.text(0.05, 0.95, f'Lower Threshold: {init_lower_threshold:.2f}, Upper Threshold: {init_upper_threshold:.2f}, Retained: {feature_counts[0]}', transform=ax1.transAxes)

    # Second plot: Histogram of Intensities with semilog-y scale
    n, bins, patches = ax2.hist(df['sum_decon_pixels'], bins=hist_bins, alpha=0.75, edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Histogram of Intensities; ' + str(bit_id))

    # Vertical lines on the histogram plot
    vline_hist_lower = ax2.axvline(x=init_lower_threshold, color='r', linestyle='--')
    vline_hist_upper = ax2.axvline(x=init_upper_threshold, color='b', linestyle='--')

    # Third plot: Scatter plot of [tile_y, tile_x] colored by ['sum_decon_pixels']
    scatter = ax3.scatter(df['tile_y_px'], df['tile_x_px'], c=df[data_to_use], cmap='viridis', s=1)
    ax3.set_xlabel('Tile Y')
    ax3.set_ylabel('Tile X')
    ax3.set_title('Tile Coordinates with Intensity Colormap')

    # Slider for lower threshold
    axcolor = 'lightgoldenrodyellow'
    axthresh_lower = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
    sthresh_lower = Slider(axthresh_lower, 'Lower Threshold', min_intensity, max_intensity, valinit=init_lower_threshold)

    # Slider for upper threshold
    axthresh_upper = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
    sthresh_upper = Slider(axthresh_upper, 'Upper Threshold', min_intensity, max_intensity, valinit=init_upper_threshold)

    # Button
    axbutton = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(axbutton, 'Accept')

    # Update function
    def update(val):
        lower_threshold = sthresh_lower.val
        upper_threshold = sthresh_upper.val
        retained = len(df[(df[data_to_use] > lower_threshold) & (df[data_to_use] < upper_threshold)])
        vline_lower.set_xdata(lower_threshold)
        vline_upper.set_xdata(upper_threshold)
        vline_hist_lower.set_xdata(lower_threshold)
        vline_hist_upper.set_xdata(upper_threshold)
        text.set_text(f'Lower Threshold: {lower_threshold:.2f}, Upper Threshold: {upper_threshold:.2f}, Retained: {retained}')
        filtered_df = df[(df[data_to_use] > lower_threshold) & (df[data_to_use] < upper_threshold)]
        
        scatter.set_offsets(np.c_[filtered_df['tile_y_px'], filtered_df['tile_x_px']])
        scatter.set_array(filtered_df[data_to_use])
        
        # Redraw only the updated parts
        ax1.draw_artist(vline_lower)
        ax1.draw_artist(vline_upper)
        ax1.draw_artist(text)
        ax2.draw_artist(vline_hist_lower)
        ax2.draw_artist(vline_hist_upper)
        ax3.draw_artist(scatter)
        
        fig.canvas.draw_idle()

    sthresh_lower.on_changed(update)
    sthresh_upper.on_changed(update)

    # Accept button callback
    def accept(event):
        global accepted_lower_threshold, accepted_upper_threshold
        accepted_lower_threshold = sthresh_lower.val
        accepted_upper_threshold = sthresh_upper.val
        plt.close()

    button.on_clicked(accept)

    plt.show()
    
    del df
    
    for tile_id in tile_ids:
        tile_df = pd.read_parquet(ufish_path / tile_id / Path(bit_file_path))
        tile_df['use_spot'] = np.where((tile_df[data_to_use] > accepted_lower_threshold) & (tile_df[data_to_use] < accepted_upper_threshold), 1, 0)                      
        tile_df.to_parquet(ufish_path / tile_id / Path(bit_file_path))
        del tile_df
