import numpy as np
import random 
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CircleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_image_paths  = glob.glob(root_dir + "/input/*jpg")
        self.output_image_paths = glob.glob(root_dir + "/output/*jpg")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, idx):
      
        
        #Read Noisy Image 
        img_input  = self.transform(cv.imread(self.input_image_paths[idx],cv.IMREAD_GRAYSCALE))
        #Read Output Image 
        img_output = self.transform(cv.imread(self.output_image_paths[idx],cv.IMREAD_GRAYSCALE))


        return img_input,img_output



"""
#Test code
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
dataset = CircleDataset('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/dataset/train/', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for i in dataloader:

    img_inp,img_out = i
    print(img_inp.shape,img_out.shape)

"""