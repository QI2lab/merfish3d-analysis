import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

data_path = Path('/mnt/data/qi2lab/20240317_OB_MERFISH_7/processed_v2')
ufish_path = data_path / Path('ufish_localizations')
tile_ids = [entry.name for entry in ufish_path.iterdir() if entry.is_dir()]

temp_df = []
for tile_id in tile_ids:
    temp_df.append(pd.read_parquet(ufish_path / tile_id / Path("bit17.parquet")))
    
df = pd.concat(temp_df, ignore_index=True)

# Generate thresholds
min_intensity = df['sum_decon_pixels'].min()
max_intensity = df['sum_decon_pixels'].max()
thresholds = np.linspace(min_intensity, max_intensity, 10000)

# Calculate number of features retained for each threshold
feature_counts = [len(df[df['sum_decon_pixels'] > threshold]) for threshold in thresholds]

# Plotting
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
l, = plt.plot(thresholds, feature_counts, lw=2)
plt.xlabel('Intensity Threshold')
plt.ylabel('Number of Features Retained')
plt.title('Feature Retention vs. Intensity Threshold')

# Initial threshold
init_threshold = min_intensity

# Vertical line and text
vline = ax.axvline(x=init_threshold, color='r', linestyle='--')
text = ax.text(0.05, 0.95, f'Threshold: {init_threshold:.2f}, Retained: {feature_counts[0]}', transform=ax.transAxes)

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
    text.set_text(f'Threshold: {threshold:.2f}, Retained: {retained}')
    fig.canvas.draw_idle()

sthresh.on_changed(update)

# Accept button callback
def accept(event):
    global accepted_threshold
    accepted_threshold = sthresh.val
    print(f'Accepted threshold: {accepted_threshold}')
    plt.close()

button.on_clicked(accept)

plt.show()

# After closing the plot, filter the dataframe based on the accepted threshold
filtered_df = df[df['sum_decon_pixels'] > accepted_threshold]