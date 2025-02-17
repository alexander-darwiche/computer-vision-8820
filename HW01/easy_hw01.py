import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# Read the file
with open('img/comb.img', "rb") as f:
    f.seek(512)
    image_data = np.frombuffer(f.read(), dtype=np.uint8).copy()

# Reshape into 2D array
image = image_data.reshape((512, 512))

# Threshold the image with T = 128 to create a binary image BT
T = 128
_, binary_image = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)

# Optionally, apply morphological operations to separate connected components further
# This step can help if components are touching or close together
kernel = np.ones((3, 3), np.uint8)
binary_image = cv2.dilate(binary_image, kernel, iterations=1)  # Dilate slightly to separate components

# Label connected components in the binary image
labeled_image, num_labels = label(binary_image, connectivity=2, return_num=True)

# Get properties of the components using skimage's regionprops
regions = regionprops(labeled_image)

# Initialize list to store component descriptions
component_descriptions = []

# Process each component
for region in regions:
    # Component Size (area)
    area = region.area
    
    # Centroid
    centroid = region.centroid
    
    # Bounding Box
    minr, minc, maxr, maxc = region.bbox
    
    # Orientation (angle of the major axis)
    orientation = region.orientation
    
    # Eccentricity (measure of elongation)
    eccentricity = region.eccentricity
    
    # Perimeter (using the contour, calculate perimeter using arcLength)
    contour = np.array(region.coords)
    perimeter = cv2.arcLength(contour, True)
    
    # Compactness (P^2 / A)
    compactness = (perimeter ** 2) / area
    
    # Append the description of the component
    component_descriptions.append({
        'size': area,
        'centroid': centroid,
        'bbox': (minr, minc, maxr, maxc),
        'orientation': orientation,
        'eccentricity': eccentricity,
        'perimeter': perimeter,
        'compactness': compactness
    })

# Print out the component descriptions
for i, desc in enumerate(component_descriptions):
    print(f"Component {i+1}:")
    print(f"  Size (Area): {desc['size']}")
    print(f"  Centroid: {desc['centroid']}")
    print(f"  Bounding Box: {desc['bbox']}")
    print(f"  Orientation: {desc['orientation']}")
    print(f"  Eccentricity: {desc['eccentricity']}")
    print(f"  Perimeter: {desc['perimeter']}")
    print(f"  Compactness: {desc['compactness']}")
    print()

# Plot the labeled components
plt.figure(figsize=(8, 8))
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.title(f"Disconnected Connected Components - {num_labels} labels")
plt.colorbar()
plt.show()
