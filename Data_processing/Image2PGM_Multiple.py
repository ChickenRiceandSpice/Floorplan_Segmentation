
from Image_Formatting import original2pgm_and_save
import pandas as pd
import PIL.Image as Image
import os

main_dir, _ = os.path.split(os.getcwd())
check = False    # Change to true if you want to see the image (either raw or clean) else keep False

if not(check):
    save_raw_images = False     # Change to False if you want to save cleaned images


    save_cleaned_images = not(save_raw_images)
    images_csv_path = main_dir + r'/Dataset_files/new_images.csv'
    df_csv = pd.read_csv(images_csv_path)
    raw_images = df_csv['Raw_Image']
    clean_images = df_csv['Cleaned_Image']

    if save_raw_images:
        saving_path = main_dir + r'/Result_evaluation/Pgm_Images'
        selected_path = main_dir + r'/Result_evaluation/FloorPlans_dataset//'
        for i in range(len(raw_images)):
            selected_image_path =  selected_path + raw_images[i]
            original2pgm_and_save(clean_images[i], selected_image_path, saving_path)



    if save_cleaned_images:
        saving_path = main_dir + r'/Result_evaluation/Cleaned_Images_pgm'
        selected_path = main_dir + r'/Result_evaluation/Cleaned_Images//'
        for i in range(len(clean_images)):
            selected_image_path = selected_path + clean_images[i]
            original2pgm_and_save(clean_images[i], selected_image_path, saving_path)


if check:

    # Uncomment this if you want to see raw image
    i = main_dir + r'/Result_evaluation/Pgm_Images/image35.pgm'

    # Uncomment this if you want to see cleaned image
    #i = main_dir + r'/Result_evaluation/Cleaned_Images_pgm/image35.pgm'
    img = Image.open(i)
    img.show()



