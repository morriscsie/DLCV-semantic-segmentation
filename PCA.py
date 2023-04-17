import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from model import CNN_base,Resnet152
import matplotlib.pyplot as plt
from dataset import P1_Train_Dataset
import random
import pandas as pd 
import seaborn as sns
from sklearn.decomposition import PCA
if __name__ == '__main__':

   
    valid_dir = "./hw1_data/p1_data/val_50/"
    #PATH = "./model/best_model_P1B.pt"
    PATH = "./15epoch_model.pt"
    input_size = 224
    valid_transform =  transforms.Compose(
        [transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
   
    valid_dataset = P1_Train_Dataset(valid_dir,transform=valid_transform)
    batch_size = 128
    val_dataloader = DataLoader(valid_dataset, batch_size, shuffle = False, num_workers = 4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Resnet152()
    ckpt = torch.load(PATH)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.model_ft
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)
    print(model)
    validate_loop = tqdm(val_dataloader,position=0,leave=False,ncols=60,colour='green')
    outputs = []
    labels = []
    with torch.no_grad():
        for i,batch in enumerate(val_dataloader):
            imgs,label = batch
            imgs = imgs.to(device)
            label = label.to(device)
            output = model(imgs)
            output = output.view(-1,2048)
            output = output.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            outputs.append(output)
            labels.append(label)
    #PCA
    x = np.concatenate(outputs,axis=0)
    y = np.concatenate(labels,axis=0)
    print(x.shape)
    print(y.shape)
    pca = PCA(n_components=2)
    x_scaled = x
    pca_features = pca.fit_transform(x_scaled)
    print('Shape before PCA: ', x_scaled.shape)
    print('Shape after PCA: ', pca_features.shape)
    pca_df = pd.DataFrame(data=pca_features,columns=['PC1', 'PC2'])
    pca_df['target'] = y
    pca_df.head()
    sns.set()
    sns.lmplot(x='PC1',y='PC2',data=pca_df,hue='target',height=20, aspect=0.8,fit_reg=False,legend=True)
    plt.title('PCA Graph')
    plt.show()
    plt.savefig("./PCA.jpg")
              
      

