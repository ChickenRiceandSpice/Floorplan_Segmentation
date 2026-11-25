import os
import matplotlib.pyplot as plt

class Selectimage:
    def __init__(self):
        pass

    def chooseimage(self):
        self.path_name = os.getcwd()
        self.pgm_name = self.path_name + '\Pgm_Images'
        self.files = os.listdir(self.pgm_name)
        if len(self.files) > 1:
            for i in range(len(self.files)):
                print(str(i + 1) + '. ' + self.files[i])
            x = input('Enter a number corresponding to the image file to be selected: ')
            if int(x) > (len(self.files)):
                print('Number out of range')
                selected_image = 'No image selected'
                return selected_image
            else:
                selected_image = self.pgm_name + '\\' + self.files[int(x) - 1]
                return selected_image


def get_path():
    dir_name = os.getcwd()
    folder_name = dir_name + '\FloorPlans_dataset\\'
    return folder_name

def retrieve_image_file(folder_name):
    files = os.listdir(folder_name)
    for i in range(len(files)):
        print(str(i+1)+ '. ' + files[i])

    x = input('Enter a number corresponding to the image file to be selected: ')
    if int(x) > (len(files)):
        print('Number out of range')
        selected_image = 'No image selected'
        return selected_image
    else:
        selected_image = files[int(x)-1]
        print('The selected image is:\n ', selected_image)
        return selected_image


def get_image_path(selected_image):
    folder_name = get_path()
    image_path = folder_name + selected_image
    return image_path


def open_pgm():
    # path_name = os.getcwd()
    # pgm_name = path_name + '\Pgm_Images'
    # files = os.listdir(pgm_name)
    # if len(files) > 1:
    #     for i in range(len(files)):
    #         print(str(i+1)+ '. ' + files[i])
    #     x = input('Enter a number corresponding to the image file to be selected: ')
    #     if int(x) > (len(files)):
    #         print('Number out of range')
    #         selected_image = 'No image selected'
    #     else:
    #         selected_image = pgm_name + '\\' + files[int(x)-1]

    select1 = Selectimage()
    with open(select1.chooseimage(), 'rb') as pgm_open:
        pgm_img = plt.imread(pgm_open)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pgm_img)
    plt.show()


