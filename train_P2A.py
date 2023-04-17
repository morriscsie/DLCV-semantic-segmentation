import torch
import torch.nn as nn
import torchvision
import torch.optim
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
import os
from model import VGG16_FCN32
from dataset import P2_Train_Dataset
from argparse import ArgumentParser
from mean_iou_evaluate import mean_iou_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
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
def main(args):
    # Seed
    seed = 94
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs("./test/",exist_ok=True)
    #hyperparameter
    batch_size = 12
    epochs = 40
    learning_rate = 1e-4
    #DataLoader
    train_path = args.train_dir
    valid_path = args.valid_dir
    PATH = args.ckpt_path
    #transform
    input_size = 512
    train_transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    valid_transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    train_dataset = P2_Train_Dataset(train_path,transform=train_transform)
    valid_dataset = P2_Train_Dataset(valid_path,transform=valid_transform)

    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=2)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = VGG16_FCN32()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loop = tqdm(train_dataloader,position=0,leave=True,ncols=60,colour='green',desc='Epoch '+str(epoch+1))
        valid_loop = tqdm(valid_dataloader,position=0,leave=True,ncols=60,colour='green')
        model.train()
        total_loss = []
        total_vloss = []
        IoU = []
        for i,batch in enumerate(train_loop):
            optimizer.zero_grad()
            images,masks,a,b = batch
            images = images.to(device)
            masks = masks.to(device)#(N,512,512)(N,d1,d2)
            outputs = model(images)#(N,7,512,512)(N,C,d1,d2)
            loss = criterion(outputs,masks)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        preds = []
        labels = []
        iou = []
        best_IOU = 0.0
        train_loss = np.mean(total_loss)
        model.eval()
        with torch.no_grad():
            for i,Batch in enumerate(valid_loop):
                Images,Masks,img_path,mask_path = Batch
                Images = Images.to(device)
                Masks = Masks.to(device)
                output = model(Images)#(N,7,512,512)
                vloss = criterion(output,Masks)
                total_vloss.append(vloss.item())
                prediction = torch.argmax(output,dim=1)#(N,512,512)
                prediction = prediction.detach().cpu().numpy()
                Masks = Masks.detach().cpu().numpy()
                preds.append(prediction)
                labels.append(Masks)
        Preds = np.concatenate(preds,axis=0)
        Labels = np.concatenate(labels,axis=0)#(257,512,512)
        IoU_score = mean_iou_score(Preds,Labels)
        if IoU_score > best_IOU:
            torch.save({
            'epoch':(epoch+1),
            'model_state_dict':model.state_dict(),
            'IoU_score':IoU_score,
            }, PATH)
            best_IOU = IoU_score
        valid_loss = np.mean(total_vloss)
        print("Epoch: {:.0f}, train_loss: {:.4f}, valid_loss: {:.4f}, IoU: {:.5f}".format((epoch+1),train_loss,valid_loss,IoU_score))
    print("best_IoU = %.4f" % best_IOU)
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=Path,
        help="Directory to the train.",
        default="./hw1_data/p2_data/train/",
    )
    parser.add_argument(
        "--valid_dir",
        type=Path,
        help="Directory to the valid.",
        default="./hw1_data/p2_data/validation/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./model/best_model_P2A.pt"
    )
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    main(args)
