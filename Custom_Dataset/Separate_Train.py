# IMPORTING FILES
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image as Image
import torch.nn as nn
from Custom_Dataset.newcustomDataset import NewCustomDataset
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

main_dir, _ = os.path.split(os.getcwd())


flag1 = False     ## Change to True if the unproccessed images needs to be checked
flag2 = False    ## Change to True if the processed (after resize and tensor transformed) images needs to be checked
flag3 = True     ## Plot a random output image after passing it into the model
flag4 = True     ## Plot variation of loss over the epochs. Recommended to keep it True


# MANUALLY SELECTED INPUTS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4
num_epochs = 150
learning_rate = 0.01
img_height = 306
img_width = 306


# TRANSFORMATIONS
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size = (img_height,img_width))])

# DATASET IMPORTING
dataset = NewCustomDataset(csv_file= main_dir + r'/Custom_Dataset/new_images.csv',
                           root_dir_raw = main_dir + r'/Result_evaluation/Pgm_Images',
                           root_dir_clean = main_dir + r'/Result_evaluation/Cleaned_Images_pgm',
                           transform = transform)


train_data, val_data, test_data = torch.utils.data.random_split(dataset, [42,4,8])

# TRAINING AND TESTING DATALOADERS
train_loader = torch.utils.data.DataLoader(dataset= train_data, batch_size = batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset= val_data, batch_size = batch_size, shuffle= True)
test_loader = torch.utils.data.DataLoader(dataset=test_data , batch_size = batch_size, shuffle=False)


## Prints the raw image
if flag1:
    for i,(raw_image,clean_image) in enumerate(train_loader):
        img = raw_image[0][0].detach().numpy()
        op = Image.fromarray(img)
        op.show()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3)
        #self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)

        self.unpool2 = nn.MaxUnpool2d(2)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.convtrans2 = nn.ConvTranspose2d(in_channels=32, out_channels=20, kernel_size=3)
        self.convtrans1 = nn.ConvTranspose2d(in_channels=20, out_channels=1, kernel_size=3)
        self.unnorm2 = nn.BatchNorm2d(20)

    def forward(self,x):
        x, ind1 = self.pool1(F.relu(self.conv1(x)))
        x, ind2 = self.pool2(F.relu(self.conv2(x)))

        x = self.unpool2(x, ind2)
        x = self.convtrans2(x)
        x = self.unpool1(x, ind1)
        x = self.unnorm2(x)
        x = self.convtrans1(x)
        return x


model = Autoencoder().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_array = []
val_array = []

## Prints the raw image before the model is trained
if flag2:
    for i,(raw_image,clean_image) in enumerate(train_loader):
        output1 = model(raw_image.to(device))

    #print(output1)
    output1 = output1[0][0].cpu().detach().numpy()
    op = Image.fromarray(output1)
    op.show()



if __name__ == '__main__':
    # RUNNING EPOCHS
    n_steps = len(train_loader)
    #for epoch in tqdm(range(num_epochs), desc = "Epoch in % ", disable = False, colour="green"):
    for epoch in range(num_epochs):
        for i, (raw_image, clean_image) in enumerate(train_loader):
            raw_image = raw_image.to(device)
            clean_image = clean_image.to(device)

            # FORWARD PASS
            model.train()
            output = model(raw_image)
            loss = criterion(output-clean_image, clean_image-clean_image)
            loss_array.append(loss.item())

            # BACKWARD PASS, WEIGHT UPDATE
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # Validation Loss Check
            model.eval()
            with torch.no_grad():
                for j, (raw_image, clean_image) in enumerate(val_loader):
                    raw_image = raw_image.to(device)
                    clean_image = clean_image.to(device)
                    output = model(raw_image)
                    val_loss = criterion(output - clean_image, clean_image - clean_image)

                val_array.append(val_loss.item())

            if i % 20 == 0:
                print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')





    ### Prints the raw image passed through the model after it has been trained
    if flag3:
        for i,(raw_image,clean_image) in enumerate(test_loader):
            #print(raw_image[0][0].detach().numpy())
            output1 = model(raw_image.to(device))


        #print(output1)
        output1 = output1[0][0].cpu().detach().numpy()
        output2 = clean_image[0][0].cpu().detach().numpy()
        op = Image.fromarray(output1)
        op_2 = Image.fromarray(output2)
        op_2.show()
        op.show()

    if input('Should the model be saved') in ('Yes', 'y', 'YES', 'yes'):
        checkpoint = input('Provide a name to save checkpoint  ') + '.pth'
        torch.save(model.state_dict(), main_dir + '/Training and Testing/Checkpoints/' + checkpoint)
        print(f'The model saved as {checkpoint}')


    if flag4:
        epoch_array = np.arange(1, num_epochs*n_steps+1)
        plt.plot(epoch_array, loss_array, "b", label = "Training Loss")
        plt.plot(epoch_array, val_array, "g" ,label = "Validation Loss")
        plt.legend()
        plt.show()