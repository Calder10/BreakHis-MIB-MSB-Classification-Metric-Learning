import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def upload_data(train_path,val_path):
    train_data = np.load(train_path, allow_pickle=True)
    val_data = np.load(val_path, allow_pickle=True)
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    x_val = val_data['arr_0']
    y_val = val_data['arr_1']
    y_train = np.reshape(y_train,-1)
    y_val = np.reshape(y_val,-1)
    return x_train,y_train,x_val,y_val

def optimal_k_knn(x_train,y_train,x_val,y_val):
    k_values = list(range(3,22,2))
    acc = []
    for k in k_values:
        clf = KNeighborsClassifier(k)
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_val)
        a = accuracy_score(y_val,y_pred)
        acc.append(a)
    return acc
    
    

def plot_k_acc(avg_acc,m):
    k_values = list(range(3,22,2))
    plt.figure()
    plt.suptitle("Classification MSB %s (Embedding: Triplet Net) \n KNN: k vs  Accuracy" %(m),fontsize=16)
    plt.plot(k_values,avg_acc,marker = 'o')
    plt.xticks(k_values)
    plt.xlabel("k")
    plt.ylabel("avg. accuracy")
    plt.tight_layout()
    plt.show()
    
    


def run_folds():
    magn = ['40X','100X','200X','400X']
    for m in magn:
        acc = []
        for f in range(1,6):
            print("**********************************")
            print("Magnification Factor %s -  Fold %d" %(m,f))
            train_path = "../embeddings/"+str(m)+"/train/train_emb_f"+str(f)+".npz"
            val_path = "../embeddings/"+str(m)+"/test/test_emb_f"+str(f)+".npz"
            x_train,y_train,x_val,y_val = upload_data(train_path,val_path)
            a = optimal_k_knn(x_train,y_train,x_val,y_val)
            acc.append(a)
        
        acc = np.array(acc)
        avg_acc = np.mean(acc,axis=0)
        plot_k_acc(avg_acc,m)

if __name__ == '__main__':
    run_folds()