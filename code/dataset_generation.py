import numpy as np
import random 
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv


#Specifications 
width = 128
height = 128
radius = 32

root_dir = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/dataset/{}/"


#Creates Circle 
def createCircle(width,height , rad ):
  w = random.randint(1, width)
  h = random.randint(1, height)
  center = [int(w), int(h)]
  radius = rad

  Y, X = np.ogrid[:height, :width]
  dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

  mask = dist_from_center <= radius

  return mask

#Adds Circle to Image
def addCircle(test_image):
    m = createCircle(width = width, height = height , rad = radius )
    masked_img = test_image.copy()
    masked_img[m] = 255
    return masked_img

#Adds Salt and Pepper Noise to image
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output



#Save images 
def save_images(save_path,total_images):

    for i in range(total_images):

        img = np.zeros([width,height],dtype=np.uint8)
        img.fill(0)

        #for j in range(1):
        img = addCircle(test_image=img)

        cv.imwrite(save_path + f"/output/{i}.jpg",img)
        img_sp = sp_noise(img,0.1)
        cv.imwrite(save_path + f"/input/{i}.jpg",img_sp)

    #plt.imshow(img_sp)
    #plt.show()



#Save Images 
save_images(root_dir.format("train"),1000)
save_images(root_dir.format("val"),100)
save_images(root_dir.format("test"),100)


