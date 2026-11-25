import os
from PIL import Image
import numpy as np
import math
from Data_processing import Path_detection

# f = r'C:\Users\Yaash Rane\PycharmProjects\Image Detection\Result_evaluation\Pgm_Images\127_12thst_1fl.pgm'

def pixel2array():
    select2 = Path_detection.Selectimage
    img_path = select2.chooseimage(select2)
    img = Image.open(img_path)
    img_size = img.size
    print(f'The image size of the selected image is {img_size}')
    pixels = img.load()

    x_array = np.arange(0, math.ceil(img_size[0]),1)
    y_array = np.arange(0, math.ceil(img_size[1]), 1)
    pix_array = np.zeros((x_array.size, y_array.size,3), dtype = 'uint8')


    for i in x_array:
        for j in y_array:
            pix_array[i,j] = [i,j,pixels[i,j]]

    return pix_array

# data = np.zeros( (512,512,3), dtype=np.uint8)
# data[256,256] = [255,150,122]
# plt.imshow(data, interpolation='nearest')
# plt.show()
