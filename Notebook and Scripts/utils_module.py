import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from config_module import CFG as config


def segrigate_image(input_path):
    """
    Segregates the images and masks from the given input path.

    """
    image = []
    image_mask = []
    for x in sorted(os.listdir(input_path)):
        if x.endswith("png"):
            if "mask" in x:   
                image_mask.append(x)
            else:
                image.append(x)
    # print(f"Images: {len(image)}, masks: {len(image_mask)}")
    return image, image_mask


def prepare_plot(origImage, origMask,groundTruthLabel, predMask, PredictedLabel):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title(f"Original Mask , label: {groundTruthLabel} " )
    ax[2].set_title(f"Predicted Mask , label: {PredictedLabel} ")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()
    return None


def plot_training_loss(history):
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["valid_loss"], label="valid_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.path_plot)
    # serialize the model to disk
    return None

def plot_training_Accuracy(history):
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_accuracy"], label="train_accuracy")
    plt.plot(history["valid_accuracy"], label="valid_accuracy")
    plt.title("Training accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.path_plot)
    # serialize the model to disk
    return None

