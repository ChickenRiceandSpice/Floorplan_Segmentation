# IMPORTING FILES
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image as Image
import torch.nn as nn
from newcustomDataset import NewCustomDataset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from time import sleep

main_dir, _ = os.path.split(os.getcwd())

flag1 = False     ## Change to True if the unproccessed images needs to be checked
flag2 = False     ## Change to True if the processed (after resize and tensor transformed) images needs to be checked
flag3 = True     ## Plot a random output image after passing it into the model
flag4 = True     ## Plot variation of loss over the epochs. Recommended to keep it True

if flag1:
    i_raw = main_dir + r'/Result_evaluation/Pgm_Images/image1.pgm'
    i_clean = main_dir + r'/Result_evaluation/Cleaned_Images_pgm/image1.pgm'
    img_raw = Image.open(i_raw)
    img_clean = Image.open(i_clean)
    img_raw.show()
    img_clean.show()
    exit()

# MANUALLY SELECTED INPUTS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4
num_epochs = 400
learning_rate = 0.01
img_height = 406
img_width = 406

# TRANSFORMATIONS
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size = (img_height,img_width))])

# DATASET IMPORTING
dataset = NewCustomDataset(csv_file= 'new_images.csv',
                           root_dir_raw = main_dir + r'/Result_evaluation/Pgm_Images',
                           root_dir_clean = main_dir + r'/Result_evaluation/Cleaned_Images_pgm',
                           transform = transform)

train_data, test_data = torch.utils.data.random_split(dataset, [36,4])

# TRAINING AND TESTING DATALOADERS
train_loader = torch.utils.data.DataLoader(dataset= train_data, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data , batch_size = batch_size, shuffle=False)


# DATA VALIDATION
if flag2:
    data = iter(train_loader)
    raw_image, clean_image = next(data)
    print(raw_image.shape)
    print(clean_image.shape)

    raw_image = raw_image[0][0].numpy()
    clean_img = clean_image[0][0].numpy()
    img_raw = Image.fromarray(raw_image)
    img_clean = Image.fromarray(clean_img)
    img_raw.show()
    img_clean.show()
    exit()

# NEURAL NETWORK
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 5)
        # self.pool = nn.MaxPool2d(2, return_indices= True)
        # self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 5, kernel_size = 5)
        # self.pool = nn.MaxPool2d(2, return_indices= True)
        # self.conv3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5)
        # self.pool = nn.MaxPool2d(2, return_indices=True)
        # self.conv4 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5)
        # self.pool = nn.MaxPool2d(2, return_indices=True)
        # self.conv5 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5)
        # self.pool = nn.MaxPool2d(2, return_indices=True)
        # self.conv6 = nn.Conv2d(in_channels=5, out_channels=7, kernel_size=5)
        # self.pool = nn.MaxPool2d(2, return_indices=True)
        #
        # self.convtrans6 = nn.ConvTranspose2d(in_channels=7, out_channels=5, kernel_size=5)
        # self.unpool6 = nn.MaxUnpool2d(2)
        # self.convtrans5 = nn.ConvTranspose2d(in_channels=5, out_channels=5, kernel_size=5)
        # self.unpool5 = nn.MaxUnpool2d(2)
        # self.convtrans4 = nn.ConvTranspose2d(in_channels=5, out_channels=5, kernel_size=5)
        # self.unpool4 = nn.MaxUnpool2d(2)
        # self.convtrans3 = nn.ConvTranspose2d(in_channels=5, out_channels=5, kernel_size=5)
        # self.unpool3 = nn.MaxUnpool2d(2)
        # self.convtrans2 = nn.ConvTranspose2d(in_channels= 5, out_channels=3, kernel_size=5)
        # self.unpool2 = nn.MaxUnpool2d(2)
        # self.convtrans1 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=5)
        # self.unpool1 = nn.MaxUnpool2d(2)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)

        self.unpool2 = nn.MaxUnpool2d(2)
        self.convtrans2 = nn.ConvTranspose2d(in_channels=40, out_channels=20, kernel_size=3)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.convtrans1 = nn.ConvTranspose2d(in_channels=20, out_channels=1, kernel_size=3)

    def forward(self, x):
        # x, ind1 = self.pool(F.relu(self.conv1(x)))
        # x, ind2 = self.pool(F.relu(self.conv2(x)))
        # x, ind3 = self.pool(F.relu(self.conv3(x)))
        # x, ind4 = self.pool(F.relu(self.conv4(x)))
        # x, ind5 = self.pool(F.relu(self.conv5(x)))
        # x, ind6 = self.pool(F.relu(self.conv6(x)))
        #
        # x = self.unpool6(x, ind6)
        # x = self.convtrans6(x)
        # x = self.unpool5(x, ind5)
        # x = self.convtrans5(x)
        # x = self.unpool4(x, ind4)
        # x = self.convtrans4(x)
        # x = self.unpool3(x, ind3)
        # x = self.convtrans3(x)
        # x = self.unpool2(x, ind2)
        # x = self.convtrans2(x)
        # x = self.unpool1(x, ind1)
        # x = self.convtrans1(x)

        x, ind1 = self.pool1(F.relu(self.conv1(x)))
        x, ind2 = self.pool2(F.relu(self.conv2(x)))

        x = self.unpool2(x, ind2)
        x = self.convtrans2(x)
        x = self.unpool1(x, ind1)
        x = self.convtrans1(x)
        return x

# MODEL INSTANCE, LOSS CALCULATION AND OPTIMIZER
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_array = []

# RUNNING EPOCHS
n_steps = len(train_loader)
#for epoch in tqdm(range(num_epochs), desc = "Epoch in % ", disable = False, colour="green"):
for epoch in range(num_epochs):
    for i, (raw_image, clean_image) in enumerate(train_loader):
        raw_image = raw_image.to(device)
        clean_image = clean_image.to(device)

        # FORWARD PASS
        output = model(raw_image)
        loss = criterion(output, clean_image)
        loss_array.append(loss.item())

        # BACKWARD PASS, WEIGHT UPDATE
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if i % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], Step [{i}/{n_steps}], Loss: {loss:.4f}')


# ACCURACY CALCULATION
with torch.no_grad():

    for i, (raw_image, clean_image) in enumerate(test_loader):
        raw_image = raw_image.to(device)
        clean_image = clean_image.to(device)
        output = model(raw_image)
        loss = criterion(output, clean_image)

        print(loss)
    if flag3:
        output1 = output.cpu().detach().numpy()
        print(output.shape)
        for i in range(batch_size):
            output_img = Image.fromarray(output1[i][0])
            output_img.show()

if flag4:
    epoch_array = np.arange(1, num_epochs*n_steps+1)
    plt.plot(epoch_array, loss_array)
    plt.show()