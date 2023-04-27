import torch
import os
class CFG:
    path_ben = '../input/Dataset_BUSI_with_GT/benign_processed/'
    path_mali = '../input/Dataset_BUSI_with_GT/malignant_processed/'
    path_norm = '../input/Dataset_BUSI_with_GT/normal/'
    path_saved_models_dir = '../saved_models/'
    path_plot = "../plots/"
    TORCH_SEED = torch.manual_seed(42)
    RANDOM_STATE = 41  # Set the random seed

    # data parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 25
    
    #dataloader parameters
    NUM_EPOCHS  = 20
    NUM_WORKERS = 1
    
    # optimizer parameters
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-3
    
    # # lr schaduler parameters
    # SKD_STEP_SIZE = 20
    # SKD_GAMMA = 0.1
    
    # model parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False
    ARCHITECTURE_SEG = "Unet"
    ARCHITECTURE_CLASSIFY = "Resnet50"
    LOSS_SEG = "BCE"
    LOSS_CLASSIFY = "BCE"
    NOTES = ""
    MODEL_PATH_NAME_SEG = path_saved_models_dir + ARCHITECTURE_SEG + "_" + LOSS_SEG + "_" + NOTES + ".pth"
    MODEL_PATH_NAME_CLASSIFY = path_saved_models_dir + ARCHITECTURE_CLASSIFY + "_" + LOSS_CLASSIFY + "_" + NOTES + ".pth"

if __name__ == "__main__":
    config = CFG()

    # print configuration parameters
    for attribute_name in dir(config):
        if not attribute_name.startswith("_"):
            print(attribute_name, getattr(config, attribute_name))
