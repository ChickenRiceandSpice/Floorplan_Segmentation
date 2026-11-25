from PIL import Image, ImageChops, ImageFilter
from Data_processing import Path_detection
import matplotlib.pylab as plt
import os

def original2greyscale(selected_image_path):
    img_original = Image.open(selected_image_path)
    img_greyscale = img_original.convert('L')
    return img_original, img_greyscale

def user_requirements(img_original, img_greyscale):
    ask = input('Do you want to plot the image ')
    if ask in ['yes', 'y', 'Yes', 'YES']:
        img = input('Which image do you want to plot - original or greyscale ')
        if img == 'original':
            plot_image(img_original)
        elif img == 'greyscale':
            plot_image(img_greyscale)
        else:
            print('Please enter valid inputs')
    elif ask in ['no', 'n', 'No', 'NO']:
        print('Fine')
    elif ask not in ['yes', 'y', 'Yes', 'YES', 'no', 'n', 'No', 'NO']:
        print('Please enter valid inputs')

def plot_image(img):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap = 'Greys')
    plt.show()

def save_as_pgm(image_name,img_greyscale):
    image_name = image_name.split('.')
    pgm_name = image_name[0] + '.pgm'
    pgm_path = os.getcwd() + '\Pgm_Images\\' + pgm_name
    img_greyscale.save(pgm_path)
    return pgm_name


def original2pgm_and_save(image_name, selected_image_path, saving_path):
    _, img_greyscale = original2greyscale(selected_image_path)
    image_name = image_name.split('.')
    pgm_name = image_name[0] + '.pgm'
    pgm_path = saving_path + '/' + pgm_name
    img_greyscale.save(pgm_path)
