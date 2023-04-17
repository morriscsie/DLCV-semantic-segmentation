import os
import shutil
def organize_data(path):
    for i in range(50):
        os.makedirs(path+str(i)+"/",exist_ok=True)
    for filename in os.listdir(path):
        # checking if it is a file
        if os.path.isfile(os.path.join(path,filename)):
            Class = filename.split("_")[0]
            ID = filename.split("_")[1]
            original = os.path.join(path,filename)
            target = os.path.join(path,Class,ID)
            shutil.move(original, target)
if __name__=='__main__':
    #organize training data
    train_path = "./hw1_data/p1_data/train_50/"
    val_path = "./hw1_data/p1_data/val_50/"
    organize_data(train_path)
    organize_data(val_path)

  