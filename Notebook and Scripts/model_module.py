import os
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

from sklearn.metrics import accuracy_score
import utils_module
import data_module
from config_module import CFG as config
import torch
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import functional as F
from torchvision.transforms import CenterCrop


import segmentation_models_pytorch as smp
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights


class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        self.encBlocks = ModuleList([Block(channels[i], channels[i + 1])for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
        
    def forward(self, x):
        blockOutputs = []
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs

class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x
    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures

class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16), 
                 nbClasses=1, retainDim=True, outSize=(config.IMAGE_SIZE)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
            encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        # return the segmentation map
        return map
    
    
unet_smp = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
for name, p in unet_smp.named_parameters():
    if "encoder" in name:
        p.requires_grad = False


class Res_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in resnet.parameters():
            param.requires_grad=False
        resnet.fc = torch.nn.Linear(in_features=2048 , out_features=3, bias=True)
        self.resnet = resnet
        
    def forward(self, x):
        """
        Forward pass of the model
        """
        x = self.resnet(x)
        return x        
    
    
    
def train_segmentation_model(model, loss_fn, optimizer, epochs, train_dl, valid_dl):
    
    trainSteps = len(train_dl.dataset) // config.BATCH_SIZE
    validSteps = len(valid_dl.dataset) // config.BATCH_SIZE
    history = {"train_loss": [], "valid_loss": []}
    print("[INFO] training the network...")
    startTime = time.time()
    
    for e in tqdm(range(epochs)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValidLoss = 0
        # loop over the training set
        for i, (x,_, y) in enumerate(train_dl):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            
            loss = loss_fn(pred, y)
            
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x,_, y) in valid_dl:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValidLoss += loss_fn(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValidLoss = totalValidLoss / validSteps
        # update our training history
        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["valid_loss"].append(avgValidLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Valid loss: {:.4f}".format(avgTrainLoss, avgValidLoss))

    # display the total time needed to perform the training
    endTime = time.time()
    torch.save(model, config.MODEL_PATH_NAME_SEG)
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    return history

def train_classifier(model, loss_fn, optimizer, epochs, train_dl, valid_dl):
    trainSteps = len(train_dl.dataset) // config.BATCH_SIZE
    validSteps = len(valid_dl.dataset) // config.BATCH_SIZE
    history = {"train_loss": [], "valid_loss": [], "train_accuracy":[], "valid_accuracy": []}
    print("[INFO] training the network...")
    startTime = time.time()
    
    for e in tqdm(range(epochs)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValidLoss = 0
        totalTrainAcc = 0
        totalValidAcc = 0
        # loop over the training set
        for i, (x,y, _) in enumerate(train_dl):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            
            loss = loss_fn(pred, y)
            acc = accuracy_score(np.argmax(pred.detach().numpy(), axis = 1), np.argmax(y.detach().numpy(), axis = 1))
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
            totalTrainAcc += acc.item()
            
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x,y, _) in valid_dl:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValidLoss += loss_fn(pred, y)
                acc = accuracy_score(np.argmax(pred.detach().numpy(), axis = 1), np.argmax(y.detach().numpy(), axis = 1))
                totalValidAcc+=acc.item()
                
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValidLoss = totalValidLoss / validSteps
        avgTrainAcc = totalTrainAcc / (trainSteps*3)
        avgValidAcc = totalValidAcc / (validSteps*3)
        # update our training history
        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["valid_loss"].append(avgValidLoss.cpu().detach().numpy())
        history["train_accuracy"].append(avgTrainAcc)
        history["valid_accuracy"].append(avgValidAcc)
        
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Train Acc: {:.3f}, Valid loss: {:.4f}, Valid Acc: {:.3f}".format(avgTrainLoss,avgTrainAcc, avgValidLoss,avgValidAcc))

    # display the total time needed to perform the training
    endTime = time.time()
    torch.save(model, config.MODEL_PATH_NAME_CLASSIFY)
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    return history


def make_predictions(model_seg, model_classify, idx):
    # set model to evaluation mode
    model_seg.eval()
    model_classify.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(data_module.df_test["image_path"][idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (128, 128))
        orig = image.copy()
        # find the filename and generate the path to ground truth
        # mask

        groundTruthPath = data_module.df_test["mask_path"][idx]
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMask = cv2.resize(gtMask, (224, 224))
        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model_seg(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > .5) * 255
        predMask = predMask.astype(np.uint8)
        
        #make predictions, collect groung truth and predicted label
        groundTruthLabel = data_module.df_test.loc[idx,["normal", "benign", "malignant"]].values
        groundTruthLabel = ["normal", "benign", "malignant"][np.argmax(groundTruthLabel)]
         
        preds = model_classify(image)
        PredictedLabel = ["normal", "benign", "malignant"][np.argmax(preds)]
        
        # prepare a plot for visualization
        
        utils_module.prepare_plot(orig, gtMask, groundTruthLabel, predMask, PredictedLabel)
        

if __name__ == "__main__":
    model = UNet()
    inp = torch.rand((config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    out = model(inp)
    print(out.shape)
    print(F"MODEL OUTPUT: {out}")