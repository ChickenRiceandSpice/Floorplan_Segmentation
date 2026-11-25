import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F

weights = torch.tensor([[-1., 0., 1.],
                       [-20.,0.,20.],
                       [-1.,0.,1.]])

weights = weights.view((1,1,3,3), dtype = torch.float32)
print(weights.type())
transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((400,400))])
img_1 = Image.open('/home/gen79net/Downloads/Image Detection/Result_evaluation/Pgm_Images/image3.pgm')
img_1 = transforms(img_1)*255
img_1 = img_1.view((1,1,400,400), dtype = torch.float32)
img_changed = F.conv2d(img_1, weights, stride = 1)

img_changed.show()


