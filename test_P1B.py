import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from model import Resnet152
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from dataset import P1_Test_Dataset
def main(args):  
    input_size = 224
    test_transform =  transforms.Compose(
        [transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    test_dataset = P1_Test_Dataset(args.test_dir,transform=test_transform)
    batch_size = 1
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False, num_workers = 4)#pin_memory = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    model = Resnet152()
    #load model
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    test_loop = tqdm(test_dataloader,position=0,leave=True,ncols=60,colour='green')
    Pred = []
    Filename = []
    for batch in test_loop:
        model.eval()
        with torch.no_grad():
            img,filename = batch
            img = img.to(device)
            outputs = model(img)
            _,prediction = torch.max(outputs,dim=1)
            prediction = int(prediction.detach().cpu().numpy())
            filename = filename[0]
            Pred.append(prediction)
            Filename.append(filename)
    with open(args.pred_file, "w") as f:
        f.write("filename,label\n")
        l = len(Pred)
        for i in range(l):
            f.write(f"{Filename[i]},{Pred[i]}\n")
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--test_dir",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Path to the pred file.",
        required=True
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the ckpt.pt.",
        default="./model/best_model_P1B.pt"
    )
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)

      



