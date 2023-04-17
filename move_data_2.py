import os
import shutil
def organize_data(path):
    os.makedirs(path+"sat/",exist_ok=True)
    os.makedirs(path+"mask/",exist_ok=True)
    for filename in os.listdir(path):
        # checking if it is a file
        if os.path.isfile(os.path.join(path,filename)):
            ID = filename.split("_")[0] #0000
            file = filename.split("_")[1] #sat.jpg
            fileclass = file.split(".")[0] #sat
            filetype = file.split(".")[1] #jpg
            original = os.path.join(path,filename)
            target = os.path.join(path,fileclass,ID+"."+filetype) #./hw1_data/p2_data/train/sat/0000.jpg
            shutil.move(original, target) #move
if __name__=='__main__':
    #organize training data
    train_path = "./hw1_data/p2_data/train/"
    val_path = "./hw1_data/p2_data/validation/"
    organize_data(train_path)
    organize_data(val_path)

  