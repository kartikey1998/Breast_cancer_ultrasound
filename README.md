# Brest_cancer_ultrasound

## Important: Please visit the "EDA and preprocessing.ipynb" and "model_training.ipynb" files inside Notebook and Scripts directory to see the model processing and training.

The Breast Ultrasound Dataset includes 780 images of breast ultrasounds from 600 female patients, collected in 2018. The images are in PNG format, with an average size of 500*500 pixels, and are categorized into normal, benign, and malignant classes.
This dataset can be found at https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset


Overview of the different modules:

EDA and preprocessing: 
The code is intended to prepare and segregate the images and masks for the benign, malignant, and normal classes. It then checks whether an image has a corresponding mask or not. If not, it segregates the images with and without masks. The code also detects whether an image has multiple masks or not, and if there are multiple masks, it merges them and saves the merged mask in a new directory. Finally, the function plot_random_images plots ten random images from the given input path.

config_module:
This defines a configuration class CFG for a machine learning project, where various paths, data parameters, dataloader parameters, optimizer parameters, model parameters, and other related parameters are defined as class attributes. The code then creates an object of the CFG class and prints all the configuration parameters using the built-in getattr function in Python.

Utility module:
This code has four functions for image segmentation tasks. The first one, segrigate_image, sorts image and mask filenames in separate lists. The second function, prepare_plot, creates a subplot with an original image, ground truth mask, and predicted mask. The third and fourth functions, plot_training_loss and plot_training_Accuracy, create plots of training and validation loss and accuracy, respectively. These functions are useful for visualizing and evaluating the performance of a segmentation model.

data_module:
It imports several libraries and custom modules, loads images and masks into lists, and creates a Pandas DataFrame called df_dict that contains information about each image. One-Hot Encoding is used to transform the label column into three columns. The data is split into training, validation, and test datasets using given ratios. The code defines a subclass of PyTorch's Dataset class called Brest_Cancer_Data that loads and transforms the images using PyTorch's built-in image transformation functions. These transformations normalize the images and convert them to tensors.


model_module:
The code defines classes including Block, Encoder, Decoder, and UNet for UNet architecture image segmentation. Block has two convolutional layers and ReLU activation. Encoder includes multiple Blocks and a max pooling layer. Decoder includes multiple Blocks, a transposed convolutional layer, and a crop method to match input size. UNet is the complete architecture with encoder, decoder, and regression head for classification. The forward method returns the segmentation map. unet_smp initializes a UNet instance using smp.Unet method with "resnet34" architecture and encoder weights.
The Res_Net class initializes a pre-trained ResNet50 model and replaces the last fully connected layer for 3 output features. Its forward function takes an input tensor and returns the output tensor.
The train_segmentation_model function trains a segmentation model with given parameters and returns the training history. It performs the training loop, calculates the training and validation losses, and saves the trained model.
The train_classifier function trains a classification model similar to train_segmentation_model but also calculates training and validation accuracies using scikit-learn library.
make_prediction function loads and preprocesses an image from the test dataset for semantic segmentation and classification evaluation. It resizes the image and its corresponding mask, transposes the axes, and passes it through a segmentation model to obtain a predicted mask. The predicted mask is then thresholded and passed through a classification model to obtain predicted labels. The prepare_plot function from the utils_module is then called to create a plot for visualization purposes.


Model_training Notebook:
It imports necessary modules and initializes a UNet segmentation model and a ResNet classification model. It trains the classification and segmentation models, plots their training progress, and tests them on a subset of the test dataset using the make_predictions function. The code follows a standard machine learning workflow of importing modules, initializing models, training, and testing.
