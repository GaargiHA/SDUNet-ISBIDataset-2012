import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

class MyISBIData(Dataset):
    def __init__(self, volPath, lblPath, resize=(128,128)):
        self.vols = tifffile.imread(volPath)
        self.lbls = tifffile.imread(lblPath)

        self.imgTransform = T.Compose([
            T.ToPILImage(),
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize([0.5],[0.5])
        ])

        self.maskTransform = T.Compose([
            T.ToPILImage(),
            T.Resize(resize),
            T.ToTensor(),
            T.Lambda(lambda x: (x>0.5).float())
        ])


    def __len__(self):
        return len(self.vols)

    def __getitem__(self, index):
        img = self.vols[index]
        mask = self.lbls[index]
        img = self.imgTransform(img)
        mask = self.maskTransform(mask)
        return img, mask

class WeightConv(nn.Conv2d):
    def forward(self, inputTensor):
        w = self.weight
        w = w - w.mean((1,2,3), keepdim=True)
        w = w / (w.view(w.size(0), -1).std(dim=1).view(-1,1,1,1) + 1e-5)
        return F.conv2d(inputTensor, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class DepthConv(nn.Module):
    def __init__(self, inChannel, outChannel):
        super().__init__()
        self.depthConv = WeightConv(inChannel, inChannel, 3, padding=1, groups=inChannel)
        self.pointConv = nn.Conv2d(inChannel, outChannel, 1)

    def forward(self, inputTensor):
        return self.pointConv(self.depthConv(inputTensor))

class SDUNet(nn.Module):
    def __init__(self, inChannel=1, outChannel=1):
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(inChannel, 32, 3, padding=1), nn.GroupNorm(8,32), nn.ReLU(True))
        self.encoder2 = nn.Sequential(DepthConv(32,64), nn.GroupNorm(8,64), nn.ReLU(True))
        self.encoder3 = nn.Sequential(DepthConv(64,128), nn.GroupNorm(8,128), nn.ReLU(True))
        self.middle = nn.Sequential(DepthConv(128,256), nn.GroupNorm(8,256), nn.ReLU(True), nn.Dropout2d(0.3))
        self.decoder3 = nn.Sequential(DepthConv(256,128), nn.GroupNorm(8,128), nn.ReLU(True))
        self.decoder2 = nn.Sequential(DepthConv(128,64), nn.GroupNorm(8,64), nn.ReLU(True))
        self.decoder1 = nn.Sequential(DepthConv(64,32), nn.GroupNorm(8,32), nn.ReLU(True))
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256,128,2,2)
        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.up1 = nn.ConvTranspose2d(64,32,2,2)
        self.outputLayer = nn.Conv2d(32, outChannel, 1)

    def forward(self, inputTensor):
        e1 = self.encoder1(inputTensor)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        b = self.middle(self.pool(e3))
        d3 = self.decoder3(torch.cat([self.up3(b), e3], 1))
        d2 = self.decoder2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], 1))
        return self.outputLayer(d1)

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bceLoss = nn.BCEWithLogitsLoss()

    def diceLoss(self, pred, target, eps=1e-6):
        p = torch.sigmoid(pred)
        return 1 - (2*(p*target).sum() + eps) / (p.sum() + target.sum() + eps)

    def forward(self, pred, target):
        return self.alpha * self.bceLoss(pred, target) + (1 - self.alpha) * self.diceLoss(pred, target)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
volPath = "/content/ISBI2012/train-volume.tif"
labelPath = "/content/ISBI2012/train-labels.tif"
data = MyISBIData(volPath, labelPath, resize=(128,128))
trainSize = int(0.8 * len(data))
trainSet, valSet = torch.utils.data.random_split(data, [trainSize, len(data)-trainSize])
trainLoader = DataLoader(trainSet, batch_size=4, shuffle=True)
valLoader = DataLoader(valSet, batch_size=4, shuffle=False)

model = SDUNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
lossFunction = ComboLoss()
bestVal = float('inf')

for epoch in range(50):
    model.train()
    trainLoss = 0
    for images, masks in trainLoader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = lossFunction(model(images), masks)
        loss.backward()
        optimizer.step()
        trainLoss += loss.item()

    model.eval()
    valLoss = 0
    with torch.no_grad():
        for images, masks in valLoader:
            images, masks = images.to(device), masks.to(device)
            valLoss += lossFunction(model(images), masks).item()

    avgTrain = trainLoss / len(trainLoader)
    avgVal = valLoss / len(valLoader)
    print("Epoch:", epoch+1, "| Train Loss:", round(avgTrain,4), "| Val Loss:", round(avgVal,4))

    if avgVal < bestVal:
        bestVal = avgVal
        torch.save(model.state_dict(), "bestSDUNet.pth")

model.load_state_dict(torch.load("bestSDUNet.pth", map_location=device))
model.eval()
plt.figure(figsize=(15,10))
samples = random.sample(range(len(valSet)), 4)
with torch.no_grad():
    for i, index in enumerate(samples):
        image, mask = valSet[index]
        predict = torch.sigmoid(model(image.unsqueeze(0).to(device))).squeeze().cpu() > 0.5
        plt.subplot(4,3,i*3+1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title('Input image data')
        plt.subplot(4,3,3 *i+2)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.subplot(4,3,i*3+3)
        plt.imshow(predict, cmap='gray')
        plt.title('Our Prediction')
plt.tight_layout()
plt.show()
