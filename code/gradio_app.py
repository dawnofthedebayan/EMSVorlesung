#from __future__ import annotations
import gradio as gr
import torch 
import requests
from PIL import Image
from torchvision import transforms
from model import autoencoder
from torch.autograd import Variable
import cv2 as cv

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

model = autoencoder().cuda()
model.load_state_dict(torch.load('./conv_autoencoder.pth'))
model.eval()

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 128, 128)
    x = torch.squeeze(x)

    return x


def predict(choice,inp):
        
    inp = inp[:,:,:1]

    if choice == "Median":
        
        pic = cv.medianBlur(inp,3)

    elif choice == "Gaussian":
        
        pic = cv.GaussianBlur(inp,(5,5),0)
    
    elif choice == "Opening":
        
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        pic = cv.morphologyEx(inp, cv.MORPH_OPEN, kernel)

    elif choice == "Closing":

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        pic = cv.morphologyEx(inp, cv.MORPH_CLOSE, kernel)


    elif choice == "Autoencoder":
        
        inp = img_transform(inp).unsqueeze(0)
    
        inp = Variable(inp).cuda()

        with torch.no_grad():

            output = model(inp)
            pic = to_img(output.cpu().data)

        pic  = pic.numpy()
       
        
    return pic


gr.Interface(fn=predict, 
             inputs=[gr.inputs.Radio(["Median", "Gaussian", "Opening","Closing","Autoencoder"]), gr.inputs.Image(type="numpy")],
             outputs="image"
             ).launch(share=True)
