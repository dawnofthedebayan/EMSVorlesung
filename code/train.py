import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datamodel import CircleDataset
import os
from model import autoencoder,autoencoder_2
if not os.path.exists('./train_img_ae_2'):
    os.mkdir('./train_img_ae_2')

if not os.path.exists('./val_img_ae_2'):
    os.mkdir('./val_img_ae_2')
if not os.path.exists('./test_img_ae_2'):
    os.mkdir('./test_img_ae_2')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 128, 128)
    return x


num_epochs = 200
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

#Initialise Dataset 
train_dataset = CircleDataset('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/dataset/train/', transform=img_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CircleDataset('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/dataset/val/', transform=img_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CircleDataset('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/EMSVorlesung/dataset/test/', transform=img_transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)



best_v_loss = 1e6
model = autoencoder_2().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
for epoch in range(num_epochs):
    total_loss = 0
    # ===================Training====================
    model.train(True)
    for data in train_loader:
        img_inp, img_out = data
        img = Variable(img_inp).cuda()
        img_out = Variable(img_out).cuda()
        # ===================forward=====================
        output = model(img)
        #print(output.shape,img_out.shape)
        loss = criterion(output, img_out)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, total_loss))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './train_img_ae_2/pred_image_{}.png'.format(epoch))
        pic = to_img(img_inp.cpu().data)
        save_image(pic, './train_img_ae_2/gt_image_{}.png'.format(epoch))

    # ===================Validation====================
    with torch.no_grad():
        model.eval()
        running_vloss = 0.0
        counter = 0
        for i, vdata in enumerate(val_loader):
            counter += 1
            img_inp, img_out = vdata
            img_inp = Variable(img_inp).cuda()
            img_out = Variable(img_out).cuda()
            output = model(img_inp)
            vloss = criterion(output, img_out)
            running_vloss += vloss

        avg_vloss = running_vloss/counter

        if avg_vloss < best_v_loss:

            
            pic = to_img(img_inp.cpu().data)
            save_image(pic, './val_img_ae_2/gt_image_{}.png'.format(epoch))

            pic = to_img(output.cpu().data)
            save_image(pic, './val_img_ae_2/pred_image_{}.png'.format(epoch))

            print("Saving model...")
            best_v_loss = avg_vloss

            print(model.encoder)
            torch.save(model.state_dict(), './conv_autoencoder_ae_2.pth')


# ===================Testing====================
model = autoencoder_2().cuda()
model.load_state_dict(torch.load('./conv_autoencoder_ae_2.pth'))
model.eval()


for i, vdata in enumerate(test_loader):
    counter += 1
    img_inp, img_out = vdata
    img_inp = Variable(img_inp).cuda()
    img_out = Variable(img_out).cuda()

    output = model(img_inp)

    pic = to_img(img_inp.cpu().data)
    save_image(pic, './test_img_ae_2/gt_image_{}.png'.format(i))

    pic = to_img(output.cpu().data)
    save_image(pic, './test_img_ae_2/pred_image_{}.png'.format(i))
    


        

