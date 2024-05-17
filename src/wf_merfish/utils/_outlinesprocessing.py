import numpy as np
import pandas as pd
from cellpose.utils import masks_to_outlines

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
    outlines_list = masks_to_outlines(label_image)
    outlines = {}
    
    for i, outline in enumerate(outlines_list):
        cell_id = i + 1  # Assign a unique identifier starting from 1
        outlines[cell_id] = outline
    
    return outlines

def create_microjson(outlines,
                    spacing: np.ndarray,
                    origin: np.ndarray,
                    affine: np.ndarray):
    features = []
    for cell_id, contour in outlines.items():
        transformed_coords = [warp_pixels_noz(point, spacing, origin, affine).tolist() for point in contour]
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
    for cell_id, contour in outlines.items():
        centroid = np.mean(contour, axis=0)
        global_centroid = warp_pixels_noz(centroid,spacing,origin,affine)
        centroids.append({"cell_id": cell_id, "centroid_y": centroid[0], "centroid_x": centroid[1]})
    
    return pd.DataFrame(centroids)