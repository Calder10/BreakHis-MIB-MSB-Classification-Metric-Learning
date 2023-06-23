import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import argparse

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fold", type=int,required=True, choices=[1,2,3,4,5], help="Fold")
    args = parser.parse_args()
    fold = str(args.fold)
    return fold

def upload_data(f):
    mapping={0: "Benign", 1:"Malignant"}
    paths=[]
    fold_path="dsfold%s.txt" %(f)
    emb_path="../embeddings/test/test_emb_f%s.npz" %(f)
    emb_path_train="../embeddings/train/train_emb_f%s.npz" %(f)
    db=open(fold_path)
    for row in db.readlines():
        columns = row.split('|')
        imgname = columns[0]
        grp = columns[3].strip()  # train or test
        if(grp=='test'):
            paths.append(imgname)
        
    emb_data=np.load(emb_path)
    x_emb=emb_data["arr_0"]
    y_emb=emb_data["arr_1"]

    aus = y_emb.tolist()
    y_emb = list(map(mapping.get, aus))
    y_emb = np.asarray(y_emb)


    emb_data_train=np.load(emb_path_train)
    x_emb_train=emb_data_train["arr_0"]
    y_emb_train=emb_data_train['arr_1']
    aus = y_emb_train.tolist()
    y_emb_train = list(map(mapping.get, aus))
    y_emb_train = np.asarray(y_emb_train)

    return x_emb_train,y_emb_train,x_emb,y_emb,paths

def plot_data(x_emb_train,y_emb_train,x_emb,y_emb,paths,f):
    hue_order = ['Benign','Malignant']
    mapping={
        1:["22549G","15687B","11520","21998AB"],
        2:["16184CD","23060CD","16456"],
        3:["19854C","12312","13412"],
        4:["29960AB","19854C","9146","18650","22549AB"],
        5:["22549AB","17901"]
    }

    mapping_patients={
        "22549G":"Benign",
        "15687B":"Malignant",
        "16184CD":"Benign",
        "19854C":"Benign",
        "29960AB":"Benign",
        "9146":"Malignant",
        "11520":"Malignant",
        "21998AB":"Benign",
        "23060CD":"Benign",
        "16456":"Malignant",
        "12312":"Malignant",
        "13412":"Malignant",
        "18650":"Malignant",
        "22549AB":"Benign",
        "17901":"Malignant"

    }

    pat=mapping[int(f)]
    embedder = umap.UMAP(random_state=42)
    emb_train=embedder.fit_transform(x_emb_train)
    emb= embedder.transform(x_emb)
    for p in pat:
        x_emb_pat,y_emb_pat=[],[]
        indicies = [i for i, x in enumerate(paths) if p in x]
        x_emb_pat = [emb[index] for index in indicies]
        y_emb_pat=[p]*len(x_emb_pat)
        x_emb_pat=np.array(x_emb_pat)
        y_emb_pat=np.array(y_emb_pat)

        title="%s - %s" %(p,mapping_patients[p])
        path="../plots/emb patients/%s/%s.png" %(f,p)
        plt.figure()
        plt.title(title)
        sns.scatterplot(x=emb_train[:,0],y=emb_train[:,1],marker="s",hue=y_emb_train,s=10,palette="Set1",hue_order=hue_order,alpha=0.2)
        sns.scatterplot(x=x_emb_pat[:,0],y=x_emb_pat[:,1],hue=y_emb_pat,marker="o",palette='Oranges',s=50)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path,dpi=300)
        


        
        


def main():
    f=parse_argument()
    x_emb_train,y_emb_train,x_emb,y_emb,paths=upload_data(f)
    plot_data(x_emb_train,y_emb_train,x_emb,y_emb,paths,f)

if __name__ == '__main__':
    main()