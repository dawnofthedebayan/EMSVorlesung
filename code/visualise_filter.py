import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
from model import autoencoder,autoencoder_2
import torch

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    print(tensor.shape)

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == "__main__":
    model = autoencoder_2()
    model.load_state_dict(torch.load('./conv_autoencoder_ae_2.pth'))
    model.eval()
    #Encoder Filter 1 
    filter = model.encoder[0].weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    #plt.show()
    plt.savefig('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/code/kernel_weights/trained_2/enc_conv_1.jpg')
    plt.clf()

    #Encoder Filter 2 
    filter = model.encoder[3].weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    #plt.show()
    plt.savefig('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/code/kernel_weights//trained_2/enc_conv_2.jpg')
    plt.clf()

    #Decoder Filter 1
    filter = model.decoder[0].weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    #plt.show()
    plt.savefig('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/code/kernel_weights//trained_2/dec_conv_1.jpg')
    plt.clf()

    #Decoder Filter 2 
    filter = model.decoder[2].weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    #plt.show()
    plt.savefig('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/code/kernel_weights//trained_2/dec_conv_2.jpg')
    plt.clf()

    #Decoder Filter 3 
    filter = model.decoder[4].weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    #plt.show()
    plt.savefig('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/code/kernel_weights//trained_2/dec_conv_3.jpg')
    plt.clf()

    print(model.decoder)