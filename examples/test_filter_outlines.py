import json
import numpy as np
import shapely.geometry
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

# modify this line
data_path = Path('/mnt/data/bartelle/20240423_ECL_24CryoA_2_PL025_restart')

# don't modify below here
def load_microjson(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    outlines = {feature['properties']['cell_id']: np.array(feature['geometry']['coordinates'][0]) for feature in data['features']}
    return outlines

# Load the outlines from the JSON file
dataset_path = data_path / Path("processed_v2")
outlines_path = dataset_path / Path("segmentation") / Path("cellpose") / Path("cell_outlines.json")
outlines = load_microjson(outlines_path)

# Initialize lists to store the polygons, areas, and aspect ratios
polygons = []
areas = []
aspect_ratios = []

for polygon_coords in outlines.values():
    polygon = shapely.geometry.Polygon(polygon_coords)
    
    if polygon.is_valid:
        area = polygon.area
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        aspect_ratio = max(width / height, height / width)
        
        polygons.append(polygon)
        areas.append(area)
        aspect_ratios.append(aspect_ratio)

# Initialize the figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)

# Set the initial plot limits
min_area, max_area = min(areas), max(areas)
min_aspect_ratio, max_aspect_ratio = min(aspect_ratios), max(aspect_ratios)

# Initial plot
lines = []
for polygon, area, aspect_ratio in zip(polygons, areas, aspect_ratios):
    x, y = polygon.exterior.xy
    color = 'black' if area >= min_area and aspect_ratio <= max_aspect_ratio else 'red'
    line, = ax.plot(x, y, color=color)
    lines.append(line)

def update_plot(min_area, max_aspect_ratio):
    for line, polygon, area, aspect_ratio in zip(lines, polygons, areas, aspect_ratios):
        color = 'black' if area >= min_area and aspect_ratio <= max_aspect_ratio else 'red'
        if line.get_color() != color:
            line.set_color(color)
    ax.figure.canvas.draw_idle()

# Create sliders
ax_area = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_aspect_ratio = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_area = Slider(ax_area, 'Min Area', min_area, max_area, valinit=min_area)
slider_aspect_ratio = Slider(ax_aspect_ratio, 'Max Aspect Ratio', min_aspect_ratio, max_aspect_ratio, valinit=max_aspect_ratio)

slider_area.on_changed(lambda val: update_plot(slider_area.val, slider_aspect_ratio.val))
slider_aspect_ratio.on_changed(lambda val: update_plot(slider_area.val, slider_aspect_ratio.val))

# Create Accept button
ax_accept = plt.axes([0.8, 0.025, 0.1, 0.04])
button_accept = Button(ax_accept, 'Accept')

def accept(event):
    accepted_polygons = [np.array(polygon.exterior.coords) for polygon, area, aspect_ratio in zip(polygons, areas, aspect_ratios) 
                         if area >= slider_area.val and aspect_ratio <= slider_aspect_ratio.val]
    print("Accepted Polygons:", len(accepted_polygons))
    # Do something with accepted_polygons, e.g., save to file, further processing, etc.

button_accept.on_clicked(accept)

plt.show()

