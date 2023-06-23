
import numpy as np 
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import tensorflow as tf
import tensorflow_addons
import tensorflow_addons as tfa
from keras.applications.resnet import ResNet152

batch_size = 32
emb_size = 512

def upload_triplet_net():
    triplet_net = tf.keras.models.Sequential()
    resnet152=ResNet152(include_top=False,
                    input_shape=(224,224,3),
                    pooling='max',classes=None,
                    weights='imagenet')

    triplet_net.add(resnet152)
    triplet_net.add(tf.keras.layers.Flatten())
    triplet_net.add(tf.keras.layers.Dense(1024,activation ='relu'))
    triplet_net.add(tf.keras.layers.Dense(512, activation=None)) 
    triplet_net.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    triplet_net = tf.keras.models.load_model("../models/triplet_net_1.h5")

    return triplet_net


def upload_data_bh():
    print("Upload data")
    mapping={0: "BreakHis - b", 1:'BreakHis - m'}
    train_data = np.load("../embeddings/train/train_emb_f1.npz", allow_pickle=True)
    test_data = np.load("../embeddings/test/test_emb_f1.npz", allow_pickle=True)
    x_train = train_data['arr_0']
    x_test = test_data['arr_0']
    bh_labels = np.hstack((train_data['arr_1'],test_data['arr_1']))
    bh_data = np.concatenate((x_train,x_test))
    aus = bh_labels.tolist()
    bh_labels = list(map(mapping.get, aus))
    bh_labels = np.asarray(bh_labels)
    return bh_data,bh_labels

def upload_data_mhist():
    print("Upload data")
    path="../../MHIST/dataset/images"
    x,y=[],[]

    for p in os.listdir(path):
        img_path=os.path.join(path,p)
        img=np.array(Image.open(img_path).resize((224,224)))
        x.append(img)
        y.append(2)
    
    x=np.array(x)
    y=np.array(y)
    
    return x,y


def create_embedding_image_net(net, x_mhist,y_mhist):
    mapping={0: "BreakHis",2:'MHIST'}
    print("Embegging of MHIST creation..")

    mhist_data = []
    for ex in x_mhist:
        a = np.resize(ex,(224,224,3)).reshape(1,224,224,3)
        out = net.predict(a)
        mhist_data.append(out.reshape(-1))

    mhist_data = np.array(mhist_data)


    mhist_labels = list(map(mapping.get, y_mhist))
    mhist_labels = np.asarray(mhist_labels)    
    print("Done")
    return mhist_data,mhist_labels

def plot_embedding(x_bh,y_bh,x_m,y_m):
    order=["BreakHis - b",'BreakHis - m']
    embedder = umap.UMAP(random_state=42)
    emb_bh = embedder.fit_transform(x_bh)
    emb_m = embedder.fit_transform(x_m)
    plt.figure()
    plt.title("Embedding - BreakHis Dataset and MHIST")    
    sns.scatterplot(x=emb_bh[:,0],y=emb_bh[:,1],hue=y_bh,marker="s",hue_order=order,palette="Set1",s=10,alpha=0.1)
    sns.scatterplot(x=emb_m[:,0],y=emb_m[:,1],hue=y_m,marker="s",palette='Set2',s=10,alpha=0.1)
    plt.tight_layout()
    plt.savefig("../plots/bh_mhist_val.png",dpi=300)
def main():
    net = upload_triplet_net()
    bh_data, bh_labels= upload_data_bh()
    x_mhist,y_mhist=upload_data_mhist()
    mhist_data, mhist_labels = create_embedding_image_net(net,x_mhist,y_mhist)
    plot_embedding(bh_data,bh_labels,mhist_data,mhist_labels)

if __name__ == '__main__':
    main()