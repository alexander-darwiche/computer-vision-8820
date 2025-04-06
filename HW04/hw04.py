import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def show_image(img,cmap_str='gray_r'):
    norm = plt.Normalize(vmin=0, vmax=1)  # Normalize so that only positive values are highlighted
    plt.imshow(img, cmap=cmap_str,norm=norm)
    plt.show()


def show_image_2(img,cmap_str='gray_r'):
    norm = plt.Normalize(vmin=0, vmax=255)  # Normalize so that only positive values are highlighted
    plt.imshow(img, cmap=cmap_str,norm=norm)
    plt.show()

def show_image_3(img,cmap_str='gray'):
    norm = plt.Normalize(vmin=0, vmax=255)  # Normalize so that only positive values are highlighted
    plt.imshow(img, cmap=cmap_str,norm=norm)
    plt.show()

def show_image_4(img,cmap_str='gray'):
    norm = plt.Normalize(vmin=0, vmax=1)  # Normalize so that only positive values are highlighted
    plt.imshow(img, cmap=cmap_str,norm=norm)
    plt.show()

def log_kernels_2D(sigma):
    mask = np.zeros((int(sigma * 8.0 + 1), int(sigma * 8.0+1)))
    mask2 = np.zeros((int(sigma * 8.0 + 1), int(sigma * 8.0+1)))

    kernel_size = mask.shape[0]
    # Step 1: Create meshgrid for plotting
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    x, y = np.meshgrid(x, y)

    # Step 2: Apply Laplacian of Gaussian to generate Z values
    norm_factor = 1 / (2 * np.pi * sigma**4)
    z = norm_factor * (1 - (x**2 + y**2) / (2 * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # z = -sigma*((x**2 + y**2 - 2*(sigma**2))/(sigma**4)) * np.exp(-(x**2+y**2)/(2*(sigma**2)))
    mask = z
    # import pdb;pdb.set_trace()
    
    for indx, x in enumerate(range(int(-4.0*sigma), int(4.0*sigma+1))):
        for indy, y in enumerate(range(int(-4.0*sigma), int(4.0*sigma+1))):
            mask2[indx,indy] = ((x**2 + y**2 - 2*(sigma**2))/(sigma**4)) * math.exp(-(x**2+y**2)/(2*(sigma**2)))
            # mask2[indx,indy] = norm_factor * (1 - (x**2 + y**2) / (2 * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    return mask2

def pad_image(image, pad):
    width = len(image[0])
    
    # Pad top and bottom with rows of pad_value
    padded_image = [[0] * (width + 2 * pad) for _ in range(pad)]

    # Pad each row with pad_value on left and right
    for row in image:
        row = row.tolist()
        padding = [0] * pad
        padded_row = padding + row + padding
        padded_image.append(padded_row)

    # Add more rows of pad_value at the bottom
    padded_image += [[0] * (width + 2 * pad) for _ in range(pad)]

    return np.array(padded_image)


def convolve(kernel, image, edge_map):
    padding_size = len(kernel)//2
    padded_image = pad_image(image, pad=(padding_size))
    new_image = np.zeros(image.shape, dtype=np.float64)
    padded_edge_map = np.pad(edge_map, pad_width=1, mode='constant', constant_values=0)
    for row in range(len(new_image)):
        for col in range(len(new_image[row])):
            if np.max(padded_edge_map[row:row+3,col:col+3]) == 1:
                new_image[row][col] = np.sum(kernel * padded_image[row:row+padding_size*2+1,col:col+padding_size*2+1])
            else:
                new_image[row][col] = padded_image[row+padding_size+1,col+padding_size+1]
    return new_image

def create_edge_map(image, previous_em):
    edge_map = np.zeros(image.shape)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    for row in range(len(edge_map)):
        for col in range(len(edge_map[row])): 
            if previous_em[row][col] == 1:
                if (np.max(padded_image[row:row+3,col:col+3]) > 0 and np.min(padded_image[row:row+3,col:col+3]) < 0):
                    edge_map[row][col] = 1
                else:
                    edge_map[row][col] = 0
            
    return edge_map

# Parameter Definitions
file_path1 = "img/test1.img"
file_path2 = "img/test2.img"
file_path3 = "img/test3.img"

files = [file_path1,file_path2,file_path3]
width, height = 512, 512
header_size = 512
image_data = []
combos = []



# Read the file(s)
for i in range(len(files)):
    with open(files[i], "rb") as f:
        f.seek(header_size)
        image_data.append(list(f.read()))

for i in range(len(image_data)):
    image_data[i] = (np.array(image_data[i], dtype=np.float64).reshape((height,width)))

import copy
import cv2
# Create a blank black image
image = np.zeros((100, 100), dtype=np.uint8)

# Add one distinct white circle (creates a strong edge)
center = (50, 50)  # Center of the image
radius = 20
cv2.circle(image, center, radius, 255, -1)  # -1 fills the circle

# Add random noise
noise = np.random.normal(0, 25, image.shape).astype(np.int16)
image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

sigma = 5
edge_map = np.ones(image.shape)
log_edge_maps = []
log_maps = []
while sigma >= 1.0:
    mask = log_kernels_2D(sigma)
    LoG = convolve(mask, image, edge_map)
    edge_map = create_edge_map(LoG,edge_map)

    if sigma % 1 == 0:
        # log_maps.append(LoG)
        log_edge_maps.append(edge_map)
        show_image_4(edge_map)
        show_image(edge_map)

    sigma = sigma - .5
# Plot the results
fig, axes = plt.subplots(1, len(log_edge_maps), figsize=(15, 5))
for ax, log_image, sigma in zip(axes, log_edge_maps, range(len(log_edge_maps))):
    ax.imshow(log_image, cmap='gray_r')
    ax.set_title(f"σ = {sigma+1}")
    ax.axis('off')

plt.show()




# for i in range(len(image_data)):
#     sigma = 5
#     edge_map = np.ones(image_data[i].shape)
#     log_edge_maps = []
#     log_maps = []
#     while sigma >= 1.0:
#         mask = log_kernels_2D(sigma)
#         LoG = convolve(mask, image_data[i], edge_map)
#         edge_map = create_edge_map(LoG,edge_map)

#         if sigma % 1 == 0:
#             # log_maps.append(LoG)
#             log_edge_maps.append(edge_map)
#             # show_image_4(edge_map)

#         sigma = sigma - .5
#     # Plot the results
#     fig, axes = plt.subplots(1, len(log_edge_maps), figsize=(15, 5))
#     for ax, log_image, sigma in zip(axes, log_edge_maps, range(len(log_edge_maps))):
#         ax.imshow(log_image, cmap='gray_r')
#         ax.set_title(f"σ = {sigma+1}")
#         ax.axis('off')

#     plt.show()



# for i in range(len(image_data)):
#     sigma = 5
#     edge_map = np.ones(image_data[i].shape)
#     log_edge_maps = []
#     log_maps = []
#     while sigma >= 5.0:
#         mask = log_kernels_2D(sigma)
#         LoG = convolve(mask, image_data[i], edge_map)
#         edge_map = create_edge_map(LoG, edge_map)

#         if sigma % 1 == 0:
#             # log_maps.append(LoG)
#             log_edge_maps.append(edge_map)
#             # show_image_4(edge_map)

#         sigma = sigma - .5
#     # Plot the results
#     num_maps = len(log_edge_maps)

#     fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
#     if num_maps == 1:
#         axes = [axes]

#     for ax, log_image, sigma in zip(axes, log_edge_maps, range(num_maps)):
#         ax.imshow(log_image, cmap='gray_r')
#         ax.set_title(f"σ = {sigma+1}")
#         ax.axis('off')

#     plt.show()




def plot_mask(mask):
        # Create a meshgrid for the X and Y axes
    x = np.arange(mask.shape[0])
    y = np.arange(mask.shape[1])
    x, y = np.meshgrid(x, y)

    # Initialize a figure for plotting
    fig = plt.figure(figsize=(10, 7))

    # Create 3D axes
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x, y, mask, cmap='viridis', edgecolor='none')

    # Add labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z (mask values)')
    ax.set_title('3D Surface Plot of Mask')

    # Show the color bar
    fig.colorbar(surf)

    # Display the plot
    plt.show()
