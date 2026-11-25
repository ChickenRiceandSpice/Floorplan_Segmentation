import torch
from Custom_Dataset.Separate_Train import Autoencoder
from Custom_Dataset.newcustomDataset import NewCustomDataset
import torchvision.transforms as transforms
import os
import PIL.Image as Image


main_dir, _ = os.path.split(os.getcwd())
img_height = 306
img_width = 306
batch_size = 4

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size = (img_height,img_width))])

model1 = Autoencoder()
model1.load_state_dict(torch.load(r'/home/gen79net/Downloads/Image Detection/Training and Testing/Checkpoints/checkpoint_s306_k3.pth'))
model1.eval()

dataset = NewCustomDataset(csv_file= main_dir + r'/Custom_Dataset/new_images.csv',
                           root_dir_raw = main_dir + r'/Result_evaluation/Pgm_Images',
                           root_dir_clean = main_dir + r'/Result_evaluation/Cleaned_Images_pgm',
                           transform = transform)

test_data,_ = torch.utils.data.random_split(dataset,[52,2])
test_loader= torch.utils.data.DataLoader(dataset=test_data , batch_size = batch_size, shuffle=True)


for i, (raw_image,clean_image) in enumerate(test_loader):
    output1 = model1(raw_image)
    output2 = raw_image


print(output1.size())

for i in range(batch_size):

    img_cleaned = output1[i][0].detach().numpy()
    img_raw = output2[i][0].detach().numpy()

    img_cleaned = Image.fromarray(img_cleaned)
    img_raw = Image.fromarray(img_raw)
    img_cleaned.show()
    img_raw.show()

















