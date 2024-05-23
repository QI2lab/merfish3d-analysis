import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

data_path = Path('/mnt/data/bartelle/20240423_ECL_24CryoA_2_PL025_restart')
ufish_path = data_path / Path('processed_v2') / Path('ufish_localizations')
tile_ids = sorted([entry.name for entry in ufish_path.iterdir() if entry.is_dir()])

first_tile_path = ufish_path / tile_ids[0]
file_info = []
for bit_file in first_tile_path.glob('bit*.parquet'):
    full_file_name = bit_file.name
    part_before_dot = bit_file.stem
    file_info.append((full_file_name, part_before_dot))

for bit_file_path, bit_id in file_info:
    temp_df = []
    for tile_id in tile_ids:
        temp_df.append(pd.read_parquet(ufish_path / tile_id / Path(bit_file_path)))

    df = pd.concat(temp_df, ignore_index=True)
    print(df.head())

    # Generate thresholds
    min_intensity = df['sum_decon_pixels'].min()
    max_intensity = df['sum_decon_pixels'].max()
    thresholds = np.linspace(min_intensity, max_intensity, 10000)

    # Calculate number of features retained for each threshold
    feature_counts = [len(df[df['sum_decon_pixels'] > threshold]) for threshold in thresholds]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, wspace=0.3)

    # First plot: Feature Retention vs. Intensity Threshold
    l, = ax1.plot(thresholds, feature_counts, lw=2)
    ax1.set_xlabel('Intensity Threshold')
    ax1.set_ylabel('Number of Features Retained')
    ax1.set_title('Feature Retention vs. Intensity Threshold; '+str(bit_id))

    # Initial threshold
    init_threshold = min_intensity

    # Vertical line and text on the first plot
    vline = ax1.axvline(x=init_threshold, color='r', linestyle='--')
    text = ax1.text(0.05, 0.95, f'Threshold: {init_threshold:.2f}, Retained: {feature_counts[0]}', transform=ax1.transAxes)

    # Second plot: Histogram of Intensities
    hist_bins = np.linspace(min_intensity, max_intensity, 100)
    n, bins, patches = ax2.hist(df['sum_decon_pixels'], bins=hist_bins, alpha=0.75, edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Histogram of Intensities; '+str(bit_id))

    # Vertical line on the histogram plot
    vline_hist = ax2.axvline(x=init_threshold, color='r', linestyle='--')

    # Slider
    axcolor = 'lightgoldenrodyellow'
    axthresh = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
    sthresh = Slider(axthresh, 'Threshold', min_intensity, max_intensity, valinit=init_threshold)

    # Button
    axbutton = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(axbutton, 'Accept')

    # Update function
    def update(val):
        threshold = sthresh.val
        retained = len(df[df['sum_decon_pixels'] > threshold])
        vline.set_xdata(threshold)
        vline_hist.set_xdata(threshold)
        text.set_text(f'Threshold: {threshold:.2f}, Retained: {retained}')
        fig.canvas.draw_idle()

    sthresh.on_changed(update)

    # Accept button callback
    def accept(event):
        global accepted_threshold
        accepted_threshold = sthresh.val
        plt.close()

    button.on_clicked(accept)

    plt.show()
    
    del df
    
    for tile_id in tile_ids:
        tile_df = pd.read_parquet(ufish_path / tile_id / Path(bit_file_path))
        tile_df['use_spot'] = np.where(tile_df['sum_decon_pixels'] > accepted_threshold, 1, 0)                      
        tile_df.to_parquet(ufish_path / tile_id / Path(bit_file_path))
        del tile_df