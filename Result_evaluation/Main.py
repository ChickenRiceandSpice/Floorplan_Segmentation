import Data_processing.Image_Formatting as Image_Formatting
import Data_processing.Path_detection as Path_detection
import Data_processing.Read_Pixels as Read_Pixels

continue_prompt = 'Yes'
while(continue_prompt in ['yes', 'y', 'Yes', 'YES']):
    x = input('Choose the operation required: '
          '\n 1. Convert image to pgm'
          '\n 2. Open an existing pgm image'
          '\n 3. Read Pixels \n')

    if x == '1':
        folder = Path_detection.get_path()
        image_selected = Path_detection.retrieve_image_file(folder)
        image_path = Path_detection.get_image_path(image_selected)

        img_original, img_greyscale = Image_Formatting.original2greyscale(image_path)

        Image_Formatting.user_requirements(img_original, img_greyscale)
        if input('Do you want to save the image ') in ['yes', 'y', 'Yes', 'YES']:
            pgm_name = Image_Formatting.save_as_pgm(image_selected, img_greyscale)
            print('The image is saved successfully as ', pgm_name)

    if x == '2':
        Path_detection.open_pgm()
        # if input('Do you want to open the pgm image ') in ['yes', 'y', 'Yes', 'YES']:
        #     Path_detection.open_pgm()

    if x == '3':
        pix_array = Read_Pixels.pixel2array()
        try:
            prompt = input('Do you want to print the arrays ')
            if prompt in ['yes', 'y', 'Yes', 'YES']:
                print(pix_array)
        except: print('Finished')


    continue_prompt = input('Do you want to continue the program. If yes type "yes" \n ')


