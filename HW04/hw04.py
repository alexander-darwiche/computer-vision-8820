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
    
    for indx, x in enumerate(range(int(-4.0*sigma), int(4.0*sigma+1))):
        for indy, y in enumerate(range(int(-4.0*sigma), int(4.0*sigma+1))):
            mask[indx,indy] = ((x**2 + y**2 - 2*(sigma**2))/(sigma**4)) * math.exp(-(x**2+y**2)/(2*(sigma**2)))

    return mask

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

    return new_image

def create_edge_map(image, previous_edge_map):
    edge_map = np.zeros(image.shape)

    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    padded_edge_map = np.pad(previous_edge_map, pad_width=1, mode='constant', constant_values=0)

    threshold = 10
    for row in range(len(edge_map)):
        for col in range(len(edge_map[row])): 
            # If the current node is an 8-neighbor of a current edge
            if np.max(padded_edge_map[row:row+3,col:col+3]) == 1 and edge_map[row][col] == 0:

                # Next, check if this is a zero-crossing
                current_value = padded_image[row+1,col+1]
                left_value = padded_image[row+1,col]
                top_value = padded_image[row,col+1]
                # Check if zero-crossing with left-value
                if (current_value > 0 and left_value < 0) or (current_value < 0 and left_value > 0):
                    if abs(current_value) + abs(left_value) > threshold:
                        if abs(current_value) < abs(left_value):
                            edge_map[row,col] = 1    
                        else:
                            edge_map[row,col-1] = 1 

                # Check if zero-crossing with right-value
                elif (current_value > 0 and top_value < 0) or (current_value < 0 and top_value > 0):
                    if abs(current_value) + abs(top_value) > threshold:
                        if abs(current_value) < abs(top_value):
                            edge_map[row,col] = 1
                        else:
                            edge_map[row-1,col] = 1

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

# Reshape Images
for i in range(len(image_data)):
    image_data[i] = (np.array(image_data[i], dtype=np.float64).reshape((height,width)))


# for i in range(len(image_data)):

i = 0
sigma = 5
edge_map = np.ones(image_data[i].shape)
log_edge_maps = [image_data[i]]
while sigma >= 1.0:
    mask = log_kernels_2D(sigma)
    LoG = convolve(mask, image_data[i], edge_map)
    edge_map = create_edge_map(LoG,edge_map)

    if sigma % 1 == 0:
        log_edge_maps.append(edge_map)
        # show_image_4(edge_map)

    sigma = sigma - .5

n_cols = (len(log_edge_maps) + 1) // 2  # Determine number of columns based on number of images
fig, axes = plt.subplots(2, n_cols, figsize=(15, 10))

# Flatten the axes array to make indexing easier
axes = axes.flatten()

for ax, log_image, sigma in zip(axes, log_edge_maps, range(len(log_edge_maps))):
    ax.imshow(log_image, cmap='gray')
    ax.set_title(f"σ = {sigma + 1}")
    ax.axis('off')

# Hide any unused axes
for i in range(len(log_edge_maps), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()