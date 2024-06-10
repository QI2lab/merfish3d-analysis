import numpy as np
import pandas as pd
from cellpose.utils import masks_to_outlines, outlines_list

def warp_pixels_noz(pixel_space_point: np.ndarray,
            spacing: np.ndarray,
            origin: np.ndarray,
            affine: np.ndarray) -> np.ndarray:
    
    pixel_space_point_noz = np.asarray([0,pixel_space_point[0],pixel_space_point[1]])

    physical_space_point = pixel_space_point_noz * spacing + origin
    registered_space_point = (np.array(affine) @ np.array(list(physical_space_point) + [1]))[:-1]
    
    registered_space_point_noz = np.asarray([registered_space_point[1],registered_space_point[2]])
    
    return registered_space_point_noz

def extract_outlines(label_image):
    
    return outlines_list(label_image)

def create_microjson(outlines,
                    spacing: np.ndarray,
                    origin: np.ndarray,
                    affine: np.ndarray):
    
    features = []
    cell_id = 0
    for contour in outlines:
        transformed_coords = [warp_pixels_noz(point[::-1], spacing, origin, affine).tolist() for point in contour]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [transformed_coords]
            },
            "properties": {
                "cell_id": cell_id
            }
        }
        features.append(feature)
        cell_id = cell_id + 1
    
    microjson = {
        "type": "FeatureCollection",
        "axes": [
            {"name": "y", "unit": "micron"},
            {"name": "x", "unit": "micron"}
        ],
        "features": features
    }
    
    return microjson

def calculate_centroids(outlines,
                        spacing: np.ndarray,
                        origin: np.ndarray,
                        affine: np.ndarray):
    centroids = []
    cell_id = 0
    for contour in outlines:
        x_coords = contour[:, 1]
        y_coords = contour[:, 0]
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        centroid = [centroid_y,centroid_x]
        global_centroid = warp_pixels_noz(centroid,spacing,origin,affine)
        centroids.append({"cell_id": cell_id, "centroid_y": global_centroid[0], "centroid_x": global_centroid[1]})
        cell_id = cell_id + 1
    
    return pd.DataFrame(centroids)