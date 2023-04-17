import torch
import torch.nn as nn
import torchvision
import torch.optim
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import numpy as np
from model import Resnet152
from argparse import ArgumentParser
from tqdm import tqdm

import matplotlib.pyplot as plt
import random
from dataset import P1_Train_Dataset
def accuracy_caculate(outputs,labels):
    _,predictions = torch.max(outputs,dim=1)
    acc = int(torch.eq(predictions,labels).sum())
    return acc
def main(args):
    # Seed
    seed = 48
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #train and valid data directory
    train_dir = args.train_dir
    valid_dir =  args.valid_dir
    #load the train and test data
    # m = torch.load("./best_model_B.pt")
    # print(m["acc_rate"])
    input_size = 224
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    valid_transform =  transforms.Compose(
        [transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    train_dataset = P1_Train_Dataset(train_dir,transform=train_transform)
    valid_dataset = P1_Train_Dataset(valid_dir,transform=valid_transform)
    #train_dataset = ImageFolder(train_dir,transform = train_transform)
    #valid_dataset = ImageFolder(valid_dir,transform = valid_transform)
   
   
    img,label = train_dataset[100]
   
    #hyperparameter
    epochs = 15 #50
    learning_rate = 1e-3
    batch_size = 36
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 4)#pin_memory = True
    val_dataloader = DataLoader(valid_dataset, batch_size, shuffle = False, num_workers = 4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = Resnet152()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay=0,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_acc_rate = 0.0
    PATH = args.ckpt_path
    for epoch in range(epochs):
        train_loop = tqdm(train_dataloader,position=0,leave=False,ncols=60,colour='green',desc='Epoch '+str(epoch+1))
        validate_loop = tqdm(val_dataloader,position=0,leave=False,ncols=60,colour='green')
        model.train()
        total_loss = []
        total_vloss = []
        for batch in train_loop:
            imgs,labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs)
            loss = criterion(output,labels)
            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #scheduler.step()
        mean_loss = np.mean(total_loss)
        model.eval()
        length = 0
        ac = 0
        with torch.no_grad():
            for batch in val_dataloader:
                imgs,labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                output = model(imgs)
                vloss = criterion(output,labels)
                total_vloss.append(vloss.item())
                length+= labels.size(dim=0)
                ac += accuracy_caculate(output,labels)
        acc_rate = ac/length
        if acc_rate > best_acc_rate:
            torch.save({
                'epoch':(epoch+1),
                'model_state_dict':model.state_dict(),
                'acc_rate':acc_rate,
                }, PATH)
            best_acc_rate = acc_rate
        mean_vloss = np.mean(total_vloss)
        print("Epoch: {:.0f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format((epoch+1),mean_loss,mean_vloss,acc_rate))
    print("best_acc_rate %.4f" % best_acc_rate)
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=Path,
        help="Directory to the train.",
        default="./hw1_data/p1_data/train_50/",
    )
    parser.add_argument(
        "--valid_dir",
        type=Path,
        help="Directory to the valid.",
        default="./hw1_data/p1_data/val_50/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./model/best_model_P1B.pt"
    )
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    main(args)



