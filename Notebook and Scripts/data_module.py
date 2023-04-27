import utils_module
import numpy as np
import pandas as pd
import os
import torchvision
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from config_module import CFG as config
import cv2

norm_img, norm_mask = utils_module.segrigate_image(config.path_norm)

ben_img, ben_mask = utils_module.segrigate_image(config.path_ben)

mali_img, mali_mask = utils_module.segrigate_image(config.path_mali)

total_images = len(ben_img)+len(norm_img)+len(mali_img)
test_split = int(total_images*.2)
valid_split = int((total_images*.8)*.2)
train_split = total_images-valid_split-test_split


df_dict = {"image":[],
           "image_path":[],
          "label":[],
          "mask":[],
           "mask_path":[]
          }

# Separate lists for each label
norm_img_list = norm_img
ben_img_list = ben_img
mali_img_list = mali_img

norm_mask_list = norm_mask
ben_mask_list = ben_mask
mali_mask_list = mali_mask

# Add the lists to df_dict using extend
df_dict["image"].extend(norm_img_list)
df_dict["image"].extend(ben_img_list)
df_dict["image"].extend(mali_img_list)

df_dict["image_path"].extend([config.path_norm+x for x in norm_img_list])
df_dict["image_path"].extend([config.path_ben+x for x in ben_img_list])
df_dict["image_path"].extend([config.path_mali+x for x in mali_img_list])

df_dict["label"].extend([0 for x in range(len(norm_img_list))])
df_dict["label"].extend([1 for x in range(len(ben_img_list))])
df_dict["label"].extend([2 for x in range(len(mali_img_list))])

df_dict["mask"].extend(norm_mask_list)
df_dict["mask"].extend(ben_mask_list)
df_dict["mask"].extend(mali_mask_list)

df_dict["mask_path"].extend([config.path_norm+x for x in norm_mask_list])
df_dict["mask_path"].extend([config.path_ben+x for x in ben_mask_list])
df_dict["mask_path"].extend([config.path_mali+x for x in mali_mask_list])

df_main = pd.DataFrame(df_dict)
ohe = pd.get_dummies(df_main["label"].values)
ohe.columns = ["normal", "benign", "malignant"]
df_main = pd.concat([df_main, ohe], axis = 1)

df_test_1 = df_main[df_main["label"] == 0].sample(test_split//3, random_state= config.RANDOM_STATE)
df_test_2 = df_main[df_main["label"] == 1].sample(test_split//3, random_state= config.RANDOM_STATE)
df_test_3 = df_main[df_main["label"] == 2].sample(test_split//3, random_state= config.RANDOM_STATE)
df_test = pd.concat([df_test_1, df_test_2, df_test_3]).reset_index(drop = True)

df_main = df_main[~df_main["image"].isin(df_test["image"])] #updating the df_main so that samples doesnt repeats

df_valid_1 = df_main[df_main["label"] == 0].sample(valid_split//3, random_state= config.RANDOM_STATE)
df_valid_2 = df_main[df_main["label"] == 1].sample(valid_split//3, random_state= config.RANDOM_STATE)
df_valid_3 = df_main[df_main["label"] == 2].sample(valid_split//3, random_state= config.RANDOM_STATE)
df_valid = pd.concat([df_valid_1, df_valid_2, df_valid_3]).reset_index(drop = True)

df_main = df_main[~df_main["image"].isin(df_valid["image"])].reset_index(drop = True) #updating the df_main so that samples doesnt repeats


class Brest_Cancer_Data(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.normalizee = transforms.Normalize(mean=[0.485],std=[0.225])
        self.rezise_mask = transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        self.transform = transform
        self.transform_totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.df["image_path"][idx]).float()
        img = self.transform(img)
        label = self.df.loc[idx,["normal", "benign", "malignant"]].values
        label = np.array(label, dtype = "float")
        label =torch.from_numpy(label)
    
        mask = cv2.imread(self.df["mask_path"][2],cv2.IMREAD_GRAYSCALE)
        mask = self.transform_totensor(mask)
        # mask = self.normalizee(mask)
        mask = self.rezise_mask(mask)        
        return img, label, mask


train_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
                                     ])

valid_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
                                     ])


train_bcd = Brest_Cancer_Data(df_main, train_transform)
valid_bcd = Brest_Cancer_Data(df_valid, valid_transform)

train_dataloader = DataLoader(train_bcd, batch_size = config.BATCH_SIZE,pin_memory=config.PIN_MEMORY, shuffle = True)
valid_dataloader = DataLoader(valid_bcd, batch_size = config.BATCH_SIZE,pin_memory=config.PIN_MEMORY, shuffle = True)

if __name__=="__main__":
    train_bcd = Brest_Cancer_Data(df_main, train_transform)
    img, label, mask = train_bcd[15]
    print(img.shape, label.shape, mask.shape)
    print(label)
