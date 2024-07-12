import json
import numpy as np
import shapely.geometry
import shapely.ops
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
import zarr

# modify this line
data_path = Path('/mnt/data/qi2lab/20240317_OB_MERFISH_7')

# don't modify below here
def load_microjson(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    outlines = {feature['properties']['cell_id']: np.array(feature['geometry']['coordinates'][0]) for feature in data['features']}
    return outlines

def create_and_merge_polygons(outlines):
    polygons = []
    for coords in outlines.values():
        polygon = shapely.geometry.Polygon(coords)
        if polygon.is_valid:
            polygons.append(polygon)
    merged_polygon = shapely.ops.unary_union(polygons)
    return polygons, merged_polygon

def get_largest_polygon(merged_polygon):
    if isinstance(merged_polygon, shapely.geometry.Polygon):
        return merged_polygon
    elif isinstance(merged_polygon, shapely.geometry.MultiPolygon):
        largest_polygon = None
        max_area = 0
        for poly in merged_polygon.geoms:
            if poly.area > max_area:
                max_area = poly.area
                largest_polygon = poly
        return largest_polygon
    else:
        raise TypeError("Expected a Polygon or MultiPolygon")

def plot_polygons(ax, polygons, largest_polygon):
    lines = []
    for polygon in polygons:
        y, x = polygon.exterior.xy
        color = 'yellow' if polygon.equals(largest_polygon) else 'black'
        line, = ax.plot(x, y, color=color)
        lines.append(line)
    return lines

def update_plot(lines, polygons, min_area, max_aspect_ratio, largest_polygon):
    for line, polygon in zip(lines, polygons):
        area = polygon.area
        miny, minx, maxy, maxx = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        aspect_ratio = max(width / height, height / width)
        color = 'black' if area >= min_area and aspect_ratio <= max_aspect_ratio else 'red'
        line.set_color(color)
    plt.draw()

def accept_polygons(polygons, min_area, max_aspect_ratio):
    accepted_polygons = []
    for polygon in polygons:
        area = polygon.area
        miny, minx, maxy, maxx = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        aspect_ratio = max(width / height, height / width)
        if area >= min_area and aspect_ratio <= max_aspect_ratio:
            accepted_polygons.append(np.array(polygon.exterior.coords))
    print("Accepted Polygons:", len(accepted_polygons))
    # Do something with accepted_polygons, e.g., save to file, further processing, etc.


# Load the outlines from the JSON file
dataset_path = data_path / Path("processed_v2")
outlines_path = dataset_path / Path("segmentation") / Path("cellpose") / Path("cell_outlines.json")
outlines = load_microjson(outlines_path)
polygons, merged_polygon = create_and_merge_polygons(outlines)

fused_path = dataset_path / Path("fused") / Path("fused.zarr")
fused_zarr = zarr.open(fused_path,mode='r')

# polyDT_data = np.squeeze(np.max(np.squeeze(np.array(fused_zarr['fused_all_iso_zyx'][0,0,:])),axis=0))
voxel_size_zyx_um = fused_zarr['fused_all_iso_zyx'].attrs['spacing_zyx_um']
origin_zyx_um = fused_zarr['fused_all_iso_zyx'].attrs['origin_zyx_um']

# image_extent = [
#     origin_zyx_um[1] - (polyDT_data.shape[0] * voxel_size_zyx_um[1])//2,
#     origin_zyx_um[1] + (polyDT_data.shape[0] * voxel_size_zyx_um[1])//2, 
#     origin_zyx_um[2] + (polyDT_data.shape[1] * voxel_size_zyx_um[2])//2,
#     origin_zyx_um[2] - (polyDT_data.shape[1] * voxel_size_zyx_um[2])//2
# ]
# print(image_extent)

# Determine the largest polygon
largest_polygon = get_largest_polygon(merged_polygon)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)
#ax.set_axis_off()
# ax.set_ylim(image_extent[1]+200,image_extent[0]-200)
# ax.set_xlim(image_extent[3]-200,image_extent[2]+200)

# ax.imshow(polyDT_data, cmap='gray', extent=image_extent, origin='upper')
lines = plot_polygons(ax, polygons, largest_polygon)

# Determine min and max values for sliders
min_area = min(polygon.area for polygon in polygons)
max_area = max(polygon.area for polygon in polygons)
aspect_ratios = [max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]) / min(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]) for polygon in polygons]
min_aspect_ratio = min(aspect_ratios)
max_aspect_ratio = max(aspect_ratios)

# Create sliders
ax_area = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_aspect_ratio = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_area = Slider(ax_area, 'Min Area', min_area, max_area, valinit=min_area)
slider_aspect_ratio = Slider(ax_aspect_ratio, 'Max Aspect Ratio', min_aspect_ratio, max_aspect_ratio, valinit=max_aspect_ratio)

slider_area.on_changed(lambda val: update_plot(lines, polygons, slider_area.val, slider_aspect_ratio.val, largest_polygon))
slider_aspect_ratio.on_changed(lambda val: update_plot(lines, polygons, slider_area.val, slider_aspect_ratio.val, largest_polygon))

# Create Accept button
ax_accept = plt.axes([0.8, 0.025, 0.1, 0.04])
button_accept = Button(ax_accept, 'Accept')
button_accept.on_clicked(lambda event: accept_polygons(polygons, slider_area.val, slider_aspect_ratio.val))

plt.show()