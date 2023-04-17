import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
import os
from dataset import P2_Test_Dataset
from model import SegFormer
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
cls_voc = {0:'urban', 
           2:'rangeland',
           3:'forest',  
           6:'unknown',  
           5:'barreb land',  
           1:'Agriculture land',  
           4:'water'} 
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
    batch_size = 1
    test_transform = transforms.Compose(
        [
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
  
    test_dataset = P2_Test_Dataset(args.test_dir,transform=test_transform)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_classes = 7
    model = SegFormer()
    #load model
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    test_loop = tqdm(test_dataloader,position=0,leave=True,ncols=60,colour='green')
    for batch in test_loop:
        model.eval()
        with torch.no_grad():
            image,filename = batch
            mask = torch.zeros((batch_size,512,512),dtype=int)
            save_path = os.path.join(args.pred_dir,filename[0])
            logit  = model(image,mask)    
            prediction = torch.argmax(logit,dim=1)#(1,512,512)
            prediction = prediction.squeeze()#(512,512)
            prediction = prediction.detach().cpu().numpy()
            img = np.zeros((512, 512,3),dtype=int)
            for i in range(512):
                for j in range(512):
                    if prediction[i,j] == 0:
                        img[i, j,:] = cls_color[0]  
                    if prediction[i,j] == 1:
                        img[i, j,:] = cls_color[1] 
                    if prediction[i,j] == 2:
                        img[i, j,:] = cls_color[2] 
                    if prediction[i,j] == 3:
                        img[i, j,:] = cls_color[3]  
                    if prediction[i,j] == 4:
                        img[i, j,:] = cls_color[4]  
                    if prediction[i,j] == 5:
                        img[i, j,:] = cls_color[5] 
                    if prediction[i,j] == 6:
                        img[i, j,:] = cls_color[6] 
            im = Image.fromarray(np.uint8(img))
            im.save(save_path)
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--test_dir",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        help="Path to the pred dir.",
        required=True
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the ckpt.pt.",
        default="./model/best_model_P2B.pt"
    )
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)

      