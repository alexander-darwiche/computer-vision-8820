import numpy as np
import matplotlib.pyplot as plt


# File path
file_path = "img/comb.img"

# Image dimensions
width, height = 512, 512
header_size = 512

def show_image(img,cmap_str='gray'):
    plt.imshow(img, cmap=cmap_str)
    plt.show(block=False)

# Read the file
with open(file_path, "rb") as f:
    f.seek(header_size)
    image_data = np.frombuffer(f.read(), dtype=np.uint8).copy()

# Reshape into 2D array
image_array = image_data.reshape((height, width))

# # Display the base comb image
# show_image(image_array)

# Question 1
def find_binary_img(img):
    B_t = img.copy()
    for ind1, row in enumerate(B_t):
        for ind2, col in enumerate(row):
            B_t[ind1][ind2] = 0 if col < 128 else 1
    return B_t

b_t = find_binary_img(image_array)

# # Display the binary image B_T
# show_image(b_t)

# Question 2

def iter_connected_comps(img, filter):
    equivalence_table = dict()
    component_data = dict()
    list_equiv = []
    label = 0
    B_image = np.zeros_like(img, dtype=np.int16)
    for ind1, row in enumerate(img):
        for ind2, col in enumerate(row):
            if col == 0:
                # if col1
                if ind2 == 0:
                    left_neighbor = 0
                else:
                    left_neighbor = B_image[ind1][ind2-1]

                # if row1
                if ind1 == 0:
                    top_neighbor = 0
                else:
                    top_neighbor = B_image[ind1-1][ind2]

                if top_neighbor > 0 or left_neighbor > 0:
                    min_label = min(x for x in [top_neighbor, left_neighbor] if x > 0)
                    B_image[ind1][ind2] = min_label
                else:
                    B_image[ind1][ind2] = label
                    label += 1
                
                # Check if the neighbors are already in the equivalence table
                if top_neighbor > 0 and left_neighbor > 0 and left_neighbor != top_neighbor:
                    list_equiv.append([left_neighbor,top_neighbor])

    # delete duplicates in list
    seen_unique = list()
    seen = set()
    for i in list_equiv:
        pair = frozenset(i)
        if pair not in seen:
            seen_unique.append(i)
            seen.add(pair)
    list_equiv = seen_unique
    for i in list_equiv:

        in_items0 = len([key for key, value in equivalence_table.items() if i[0] in value])>0
        in_items1 = len([key for key, value in equivalence_table.items() if i[1] in value])>0

        # Combine lists 
        if i[0] in equivalence_table and i[1] in equivalence_table:
            if i[1] != i[0]:
                equivalence_table[i[0]] = list(set(equivalence_table[i[0]]) | set(equivalence_table[i[1]])).append(i[1])
                del equivalence_table[i[1]]

        elif in_items0 and in_items1:
            key1 = [key for key, value in equivalence_table.items() if i[0] in value][0]
            key2 = [key for key, value in equivalence_table.items() if i[1] in value][0]
            if key1 != key2:
                equivalence_table[key1] = list(set(equivalence_table[key1]) | set(equivalence_table[key2]))
                del equivalence_table[key2]

        # Add if one list exists
        elif i[0] in equivalence_table and i[1] not in equivalence_table[i[0]]:
            if in_items1:
                val = [key for key, value in equivalence_table.items() if i[1] in value][0]
                if val != i[0]:
                    equivalence_table[i[0]] = list(set(equivalence_table[i[0]]) | set(equivalence_table[val]))
                    del equivalence_table[val]
            else:
                equivalence_table[i[0]].append(i[1])

        # Add if other list exists
        elif i[1] in equivalence_table and i[0] not in equivalence_table[i[1]]:
            if in_items0:
                val = [key for key, value in equivalence_table.items() if i[0] in value][0]
                equivalence_table[i[1]] = list(set(equivalence_table[i[1]]) | set(equivalence_table[val]))
                if val != i[1]:
                    del equivalence_table[val]
            else:
                equivalence_table[i[1]].append(i[0])

        elif in_items0:
            key1 = [key for key, value in equivalence_table.items() if i[0] in value][0]
            equivalence_table[key1].append(i[1])
        
        elif in_items1:
            key1 = [key for key, value in equivalence_table.items() if i[1] in value][0]
            equivalence_table[key1].append(i[0])

        # if none exist
        else:
            equivalence_table[i[0]] = [i[1]]

    for ind1, row in enumerate(B_image):
        for ind2, col in enumerate(row):
            if col > 0:
                if col not in equivalence_table:
                    new_label = [key for key, value in equivalence_table.items() if col in value]
                    if len(new_label) > 0:
                        B_image[ind1][ind2] = new_label[0]  
     
    for ind1, row in enumerate(B_image):
        for ind2, col in enumerate(row):
            if col > 0:
                if col not in component_data:
                    size = 1
                    component_data[col] = [size,[ind1],[ind2]]
                else:
                    component_data[col][0] = component_data[col][0]+1
                    component_data[col][1].append(ind1)
                    component_data[col][2].append(ind2)

    filtered_comps = [x for x in component_data.keys() if component_data[x][0] > filter]
    

    
    def calc_centroid(component_data):
        x_total, y_total = 0, 0
        for component in (component_data):
            for ind,x in enumerate(component_data[component][1]):
                y_total += component_data[component][1][ind]
                x_total += component_data[component][2][ind]

            centroid_x = x_total/len(component_data[component][1])
            centroid_y = y_total/len(component_data[component][1])
            component_data[component].append([centroid_x,centroid_y])
        return component_data

    # Find Centroids
    component_data = calc_centroid(component_data)
    show_image(B_image)

    def find_bounding_boxes(component_data):
        for component in (component_data):
            x_min = min(component_data[component][2])
            x_max = max(component_data[component][2])
            y_min = min(component_data[component][1])
            y_max = max(component_data[component][1])
            
            component_data[component].append([x_min,x_max,y_min,y_max])
        return component_data
    
    # Find Centroids
    component_data = find_bounding_boxes(component_data)

    import pdb;pdb.set_trace()
    return B_image, equivalence_table

b_image, eql_table = iter_connected_comps(b_t,100)  

# Display the binary image B
show_image(b_image,'viridis')                
import pdb;pdb.set_trace()         

import matplotlib.colors as mcolors

# Generate a unique color for each value
unique_values = np.unique(b_image)
num_values = len(unique_values)
colors = plt.cm.get_cmap('tab10', num_values)  # Use tab10 colormap with enough unique colors

# Create a ListedColormap to assign a unique color to each value
cmap = mcolors.ListedColormap(colors(range(num_values)))
bounds = np.arange(num_values + 1) - 0.5  # Ensure discrete color separation
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot the image
plt.figure(figsize=(5, 5))
plt.imshow(b_image, cmap=cmap, norm=norm)
plt.colorbar()
plt.title("Unique Colors for Each Value")
plt.show()