import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score
import argparse

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classifier", type=str,required=True, choices=["SVM",'KNN'], help="Classifier")
    args = parser.parse_args()
    classifier = str(args.classifier)
    return classifier

def upload_data(train_path,test_path,test_path_p):
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    x_test = test_data['arr_0']
    y_test = test_data['arr_1']
    with open(test_path_p, 'rb') as f:
        test_paths = pickle.load(f)
    y_train = np.reshape(y_train,-1)
    y_test = np.reshape(y_test,-1)
    return x_train,y_train,x_test,y_test,test_paths

def classifier(x_train,y_train,x_test,ct,m):
    nk = {'40X':17,'100X':7,'200X':9,'400X':3}
    if ct =="SVM":
        clf = SVC()
    else:
        clf = KNeighborsClassifier(nk[m])
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    return y_pred

def compute_pla(y_test,y_pred,paths):
    aus = []    
    for p in paths:
        x=p.split("-")
        aus.append(x[2])
    patients = list(dict.fromkeys(aus))
    p_i=[]
    p_i=[]
    patients_dist=[]
    for p in patients:
        indicies = [i for i, x in enumerate(paths) if p in x]
        y_true = [y_test[index] for index in indicies]
        y = [y_pred[index] for index in indicies]
        acc=accuracy_score(y_true,y)
        p_i.append(acc)
        print("Patient Code: %s  Patient score: %f" %(p,acc))
    
        
    pla=sum(p_i)/len(patients)
    return pla

def ila_pla(y_test,y_pred,test_paths):
    ila = accuracy_score(y_test,y_pred)
    print("ILA=",ila)
    pla = compute_pla(y_test,y_pred,test_paths)
    print("PLA=",pla)
    return ila,pla

def save_metrics(ila,pla,ct):
    folds = range(1,5)

    ila_df = pd.DataFrame(columns=['Fold','40X','100X','200X','400X'])
    ila_df['Fold']=folds
    for key in ila:
        ila_df[key]=ila[key]
    
    pla_df = pd.DataFrame(columns=['Fold','40X','100X','200X','400X'])
    pla_df['Fold']=folds
    for key in pla:
        pla_df[key]=pla[key]
    
    avg_ila = ['Avg. ILA']
    avg_pla = ['Avg. PLA']

    print(ila_df,"\n \n")
    print(pla_df,"\n \n")


    for key in zip(ila,pla):
        a_ila = sum(ila[key[0]])/len(ila[key[0]])
        avg_ila.append(a_ila)
        a_pla = sum(pla[key[0]])/len(pla[key[0]])
        avg_pla.append(a_pla)
        

    t1 = "../results/ila_values_"+ct+".csv"
    t2 = "../results/pla_values_"+ct+".csv"
    t3 = "../results/ila_pla_avg_"+ct+".csv"
    
    avg_df = pd.DataFrame([avg_ila,avg_pla],columns=['Metric','40X','100X','200X','400X'])
    print("*****************************************************************")
    print(avg_df)
    print("*****************************************************************")


    ila_df.to_csv(t1,index=None)
    pla_df.to_csv(t2,index=None)
    avg_df.to_csv(t3,index=None)

def run_5_folds():
    ct = parse_argument()
    print("Using %s classifier" %(ct))
    magn = ['40X','100X','200X','400X']
    ila_values = {}
    pla_values = {}
    for m in magn:
        aus_ila=[]
        aus_pla=[]
        for f in range(1,5):
            print("**********************************")
            print("Magnification Factor %s -  Fold %d" %(m,f+1))
            train_path = "../embeddings/"+str(m)+"/train/train_emb_f"+str(f+1)+".npz"
            test_path = "../embeddings/"+str(m)+"/test/test_emb_f"+str(f+1)+".npz"
            test_paths_p = "../embeddings/"+str(m)+"/test/test_images_path_f"+str(f+1)
            x_train,y_train,x_test,y_test,test_paths = upload_data(train_path,test_path,test_paths_p)
            y_pred = classifier(x_train,y_train,x_test,ct,m)
            ila,pla =ila_pla(y_test,y_pred,test_paths)
            aus_ila.append(ila)
            aus_pla.append(pla)
        print("**********************************")
        ila_values[m]=aus_ila
        pla_values[m]=aus_pla
    save_metrics(ila_values,pla_values,ct)

if __name__ == '__main__':
    run_5_folds()
