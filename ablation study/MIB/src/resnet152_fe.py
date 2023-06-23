import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
from shutil import copy2,rmtree
import warnings
from datetime import timedelta
from time import perf_counter as pc
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import pickle
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using",device)

net = None
train_path = None
test_path = None
val_path= None
config = None
train_emb = None
test_emb = None
val_emb= None
patients = None
train_dataset = None
test_dataset = None
val_dataset = None
train_loader = None
test_loader = None
val_loader = None
train_images_path = None
test_images_path = None
test_images_path_p = None
train_images_path_p = None
check_train = None
check_test = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        global train_images_path, test_images_path, check_train, check_test
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        if check_train == True:
            # print(path)
            train_images_path.append(path)
        if check_test == True:
            # print(path)
            test_images_path.append(path)
        # make a new tuple that includes original and the path
        return original_tuple

def create_train_test_img_path():
    global train_images_path, test_images_path, train_dataset, test_dataset, check_train, check_test,test_images_path_p,train_images_path_p
    check_train = True
    train_images_path = []
    test_images_path = []
    print("Saving images path..")
    for i in train_dataset:
        pass
    check_train = False
    check_test = True
    for i in test_dataset:
        pass
    check_test = False
    with open(test_images_path_p, "wb") as fp:
        pickle.dump(test_images_path, fp)
    
    print("Images path saved.")


def upload_train_test_set():
    global train_path, test_path, train_dataset, test_dataset, train_loader, test_loader, val_loader,val_dataset
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = ImageFolderWithPaths(train_path, transform=transform)
    test_dataset = ImageFolderWithPaths(test_path, transform=transform)
    val_dataset = ImageFolderWithPaths(val_path, transform=transform)
    train_loader = DataLoader(train_dataset,shuffle=False,batch_size=32)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size=32)
    val_loader = DataLoader(val_dataset,shuffle=False,batch_size=32)

def delete_fold():
    path ="../fold"
    rmtree(path)

def create_folder_fold():
    os.mkdir("../fold")
    os.mkdir("../fold/train")
    os.mkdir("../fold/test")
    os.mkdir("../fold/val")
    classes = ['b','m']
    for c in classes:
        os.mkdir("../fold/train/"+c)
        os.mkdir("../fold/test/"+c)
        os.mkdir("../fold/val/"+c)




def create_fold(f):
    val_patients=[["22549AB","10926","21998CD","13412","13413","12465","22704","13200"],
                  ["22549AB","11031","21998EF","12204","10147","12465","21998AB","13200"],
                  ["22549G","10926","14134","12204","10147","12465","29315EF","13200"],
                  ["22549G","10926","25197","13412","10147","12465","29315EF","13200"],
                  ["22549G","11031","29960AB","13412","10147","12465","22704","13200"]
                 ]
    root_dir = '../BreaKHis_v1/histology_slides/breast'
    srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}
    
    path ="../src/dsfold"+str(f)+".txt"
    db = open(path)
    print("Training and test set creation....")
    for row in db.readlines():
        columns = row.split('|')
        imgname = columns[0]
        mag = columns[1]  # 40, 100, 200, or 400
        grp = columns[3].strip()  # train or test
        tumor = imgname.split('-')[0].split('_')[-1]
        srcfile = srcfiles[tumor]
        s = imgname.split('-')
        pi=s[2]
        label = s[0].split("_")[1].lower()
        sub = s[0] + '_' + s[1] + '-' + s[2]
        srcfile = srcfile % (root_dir, sub, mag, imgname)
        if(pi in val_patients[f-1]):
            copy2(srcfile,"../fold/val/"+label)
        else:
            if grp == 'train':
                dest_path = "../fold/train/"+label
                copy2(srcfile,dest_path)
            else:
                dest_path = "../fold/test/"+label
                copy2(srcfile,dest_path)
    print("Done !")

def set_paths(f):
    global train_path, test_path, train_emb, test_emb,test_images_path_p,val_path,val_emb
    train_path = "../fold/train"
    test_path = "../fold/test"
    val_path = "../fold/val"
    train_emb = "../embeddings/train/train_emb_f"+str(f)+".npz"
    test_emb = "../embeddings/test/test_emb_f"+str(f)+".npz"
    val_emb="../embeddings/test/val_emb_f"+str(f)+".npz"
    test_images_path_p = "../embeddings/test/test_images_path_f"+str(f)
    """
    print(train_path)
    print(test_path)
    print(train_emb)
    print(test_emb)
    print(test_images_path_p)
    """

def create_resnet():
    global net,device
    net = models.resnet152(pretrained=True)
    net.fc = nn.Identity()
    net.to(device)
    print(net)
    return net



def remove_folders():
    path = "../fold"
    rmtree(path)


def create_embedding():
    global net,train_loader ,test_loader, device,train_emb, test_emb,val_emb,val_loader
    x_train = []
    y_train = []
    y_test = []
    x_test = []
    x_val=[]
    y_val=[]
    print("Embedding creation....")
# since we're not training, we don't need to calculate the gradients for our outputs
    net.eval()
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            aus = labels.tolist()
            y_train+=aus
            # calculate outputs by running images through the network
            outputs = net(images).tolist()
            x_train += outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            aus = labels.tolist()
            y_test+=aus
            # calculate outputs by running images through the network
            outputs = net(images).tolist()
            x_test += outputs

    
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            aus = labels.tolist()
            y_val+=aus
            # calculate outputs by running images through the network
            outputs = net(images).tolist()
            x_val += outputs


    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_val=np.array(x_val)
    y_val=np.array(y_val)
    np.savez_compressed(train_emb,x_train,y_train)
    np.savez_compressed(test_emb,x_test,y_test)
    np.savez_compressed(val_emb,x_val,y_val)
    print("Embedding created !")

def run_5_folds():
    for i in range (1,6):
        print("Fold",i)
        create_folder_fold()
        create_fold(i)
        set_paths(i)
        upload_train_test_set()
        create_train_test_img_path()
        create_resnet()
        create_embedding()
        delete_fold()

        #os.system('clear')

if __name__ == '__main__':
    run_5_folds()