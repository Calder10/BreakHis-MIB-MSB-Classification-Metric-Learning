import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap



def upload_data(train_path,test_path):
    mapping={0: "Benign",1:'Malignant'}
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    x_test = test_data['arr_0']
    y_test = test_data['arr_1']
    y_train = np.reshape(y_train,-1)
    y_test = np.reshape(y_test,-1)
    aus = y_train.tolist()
    y_train = list(map(mapping.get, aus))
    y_train = np.asarray(y_train)
    aus = y_test.tolist()
    y_test = list(map(mapping.get, aus))
    y_test = np.asarray(y_test)
    return x_train, y_train, x_test, y_test

def plot_embedding(x_train,y_train,x_test,y_test,f,i):
    path = "../plots/emb vis/emb_vis_f%s_%s.png" %(f,i)
    order=['Benign','Malignant']
    embedder = umap.UMAP(random_state=42)
    emb_train = embedder.fit_transform(x_train)
    emb_test = embedder.transform(x_test)
    plt.figure()
    plt.suptitle("Embedding - Triplet net \n Fold " + str(f) + " - Magnification Factor " + str(i) ,fontsize = 14)
    plt.subplot(211)
    plt.title("Training set")
    sns.scatterplot(x=emb_train[:,0],y=emb_train[:,1],marker='s',palette='Set1',hue=y_train,hue_order=order,s=10,alpha=0.5)
    plt.legend()
    plt.subplot(212)
    plt.title("Test set")
    sns.scatterplot(x=emb_test[:,0],y=emb_test[:,1],hue=y_test,marker='s',palette='Set1',hue_order=order,s=10,alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path,format = 'png',dpi=300)

def create_plots():
    global train_path,test_path
    ingr = ['40X','100X','200X','400X']
    print("Saving embedings plot....")
    for i in ingr:
        prefix_train = "../embeddings/"+i+"/train"
        prefix_test = "../embeddings/"+i+"/test"
        train = os.listdir(prefix_train)
        train.sort(reverse=False)
        test = os.listdir(prefix_test)
        test.sort(reverse=False)
        for x,y in zip(train,test):
            train_path = os.path.join(prefix_train,x)
            test_path = os.path.join(prefix_test,y)
            f=(x.split('f')[1].split('.npz')[0])
            x_train, y_train, x_test, y_test = upload_data(train_path,test_path)
            plot_embedding(x_train, y_train, x_test, y_test,f,i)
    print("Done !")

if __name__ == '__main__':
    create_plots()