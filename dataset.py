import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import glob
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
voc_cls = {'urban':0, 
           'rangeland': 2,
           'forest':3,  
           'unknown':6,  
           'barreb land':5,  
           'Agriculture land':1,  
           'water':4} 
cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}
def read_masks(seg, shape):
    masks = np.zeros((shape[0], shape[1]),dtype=int)
    mask = (seg >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    masks[mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[mask == 2] = 3  # (Green: 010) Forest land 
    masks[mask == 1] = 4  # (Blue: 001) Water 
    masks[mask == 7] = 5  # (White: 111) Barren land 
    masks[mask == 0] = 6  # (Black: 000) Unknown
    return masks
class P2_Train_Dataset(Dataset):
    def __init__(self,root_dir,transform=None):
        super().__init__()
        #self.sat_dir = os.path.join(root_dir,"sat")
        #self.mask_dir = os.path.join(root_dir,"mask")
        self.img_files = glob.glob(os.path.join(root_dir,"*.jpg"))
        self.mask_files = glob.glob(os.path.join(root_dir,"*.png"))
        self.img_files.sort()
        self.mask_files.sort()
        #self.img_files.sort(key= lambda x: int(x.split("/")[-1].split(".")[0]))
        #self.mask_files.sort(key= lambda x: int(x.split("/")[-1].split(".")[0]))
        self.transform = transform
    def __len__(self)->int:
        return len(self.img_files)
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = Image.open(img_path)
        # print(img_path)
        # print(mask_path)
        mask_RGB =  np.array(Image.open(mask_path))
        Image.open(mask_path).save("./tes.jpg")
        mask = read_masks(mask_RGB, mask_RGB.shape)
        if self.transform is not None:
            image = self.transform(image)
            mask =  torch.from_numpy(mask)
 
        return image,mask, img_path.split("/")[-1].split(".")[0], str(mask_path.split("/")[-1])
class P2_Test_Dataset(Dataset):
    def __init__(self,root_dir,transform=None):
        super().__init__()
        self.img_files = glob.glob(os.path.join(root_dir,"*.jpg"))
        self.img_files.sort()
        self.transform = transform
    def __len__(self)->int:
        return len(self.img_files)
    def __getitem__(self, index):
        img_path = self.img_files[index]
        image = Image.open(img_path)
        # print(img_path)
        if self.transform is not None:
            image = self.transform(image)
 
        return image, img_path.split("/")[-1].split(".")[0]+".png"
       # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         if np.array_equal(mask_RGB[i,j,:], [0, 255, 255]):
        #             mask[i,j] = 0
        #         if np.array_equal(mask_RGB[i,j,:], [255, 255, 0]):
        #             mask[i,j] = 1
        #         if np.array_equal(mask_RGB[i,j,:], [255, 0, 255]):
        #             mask[i,j] = 2
        #         if np.array_equal(mask_RGB[i,j,:],  [0, 255, 0]):
        #             mask[i,j] = 3
        #         if np.array_equal(mask_RGB[i,j,:],  [0, 0, 255]):
        #             mask[i,j] = 4
        #         if np.array_equal(mask_RGB[i,j,:],  [255, 255, 255]):
        #             mask[i,j] = 5
        #         if np.array_equal(mask_RGB[i,j,:], [0, 0, 0]):
        #             mask[i,j] = 6
# if __name__ == '__main__':
#     #hyperparameter
#     batch_size = 32
#     epochs = 1
#     learning_rate = 1e-3

#     input_size = 512
#     #DataLoader
#     train_path = "./hw1_data/p2_data/train/"
#     valid_path = "./hw1_data/p2_data/validation/"
#     #transform
#     train_transform = transforms.Compose(
#         [
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
#     )
#     valid_transform = transforms.Compose(
#         [transforms.ToPILImage(),
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
#     )
#     train_dataset = SeqDataset(train_path,transform=train_transform)
#     valid_dataset = SeqDataset(valid_path,transform=None)
#     print(train_dataset[0])
class P1_Train_Dataset(Dataset):
    def __init__(self,img_path,transform):
        super().__init__()
        self.img_files = glob.glob(os.path.join(img_path,"*.png"))
        #print(self.img_files)
        self.transform = transform
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, index):
        img_path = self.img_files[index]
        label = int(img_path.split("/")[-1].split("_")[0])
       
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img,label
class P1_Test_Dataset(Dataset):
    def __init__(self,img_path,transform):
        super().__init__()
        self.img_files = glob.glob(os.path.join(img_path,"*.png"))
        self.transform = transform
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, index):
        img_path = self.img_files[index]
        filename = img_path.split("/")[-1]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img,filename
if __name__ == '__main__':
    train_transform = transforms.Compose(
        [#transforms.ToPILImage(),
        #transforms.RandomResizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    dataset = P1_Train_Dataset("./hw1_data/p1_data/train_50/",train_transform)
    i,l = dataset[0]
    print(l)
    
  
