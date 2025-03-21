import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show_image(img,cmap_str='gray_r'):
    norm = plt.Normalize(vmin=0, vmax=100)  # Normalize so that only positive values are highlighted
    plt.imshow(img, cmap=cmap_str,norm=norm)
    plt.show()


# Question 1: Find the Binary Image
def find_binary_img(img):
    B_t = img.copy()
    for ind1, row in enumerate(B_t):
        for ind2, col in enumerate(row):
            B_t[ind1][ind2] = 0 if col > 128 else 1
    return B_t

def find_peaks(bins):
    for i in range(len(bins)):
        if i == 0 or i == 

#### Run Code ####

# Parameter Definitions
file_path1 = "img/test1.img"
file_path2 = "img/test2.img"
file_path3 = "img/test3.img"

files = [file_path1,file_path2,file_path3]
width, height = 512, 512
header_size = 512
image_data = []

# Read the file(s)
for i in range(len(files)):
    with open(files[i], "rb") as f:
        f.seek(header_size)
        image_data.append(np.frombuffer(f.read(), dtype=np.uint8).copy())

# Reshape into 2D array
image_array = [image_data[i].reshape((height, width)) for i in range(len(files))]
image_array = [[item for sublist in image_array[i] for item in sublist] for i in range(len(image_array))]
# # Display the base comb image
# for i in range(len(files)):
#     show_image(image_array[i])


# Plotting a basic histogram
d
n, bins, patches = plt.hist(pixels, bins=50, color='skyblue', edgecolor='black')
 