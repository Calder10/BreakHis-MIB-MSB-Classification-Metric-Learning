import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score
from sklearn.metrics import roc_curve,auc
from pycm import *
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

def classifier(x_train,y_train,x_test,tc):
    if tc =='SVM':
        clf = SVC()
    else:
        clf = KNeighborsClassifier(7,p=2)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    return y_pred

def compute_pla(y_test,y_pred,paths):
    aus=[]
    for p in paths:
        x=p.split("-")
        aus.append(x[2])
    patients = list(dict.fromkeys(aus))
    p_i=[]
    for p in patients:
        indicies = [i for i, x in enumerate(paths) if p in x]
        y_true = [y_test[index] for index in indicies]
        y = [y_pred[index] for index in indicies]
        acc=accuracy_score(y_true,y)        
        p_i.append(acc)
        print("Patient Code: %s Patient score: %f" %(p,acc))
    
        
    pla=sum(p_i)/len(patients)
    return pla

def classification_metrics(y_test,y_pred,test_paths):
    ila = accuracy_score(y_test,y_pred)
    cm = ConfusionMatrix(y_test,y_pred)
    error_rate = 1-ila
    macro_precision = cm.overall_stat['PPV Macro']
    micro_precision = cm.overall_stat['PPV Micro']
    macro_recall = cm.overall_stat['TPR Macro']
    micro_recall = cm.overall_stat['TPR Micro']
    f1_macro = cm.overall_stat['F1 Macro']
    f1_micro = cm.overall_stat['F1 Micro']
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    print("ILA=",ila)
    pla = compute_pla(y_test,y_pred,test_paths)
    print("PLA=",pla)
    print("Error rate=",error_rate)
    print("Precision (Macro)=",macro_precision)
    print("Precision (Micro)=",micro_precision)
    print("Recall (Macro)=",macro_recall)
    print("Recall (Micro)=",micro_recall)
    print("F1-Score (Macro)=",f1_macro)
    print("F1-Score (Micro)=",f1_micro)
    print("AUC=",auc_score)
    return ila,pla,error_rate,macro_precision,micro_precision,macro_recall,micro_recall,f1_macro,f1_micro,auc_score

def save_metrics(metrics,tc):
    folds = range(1,6)
    columns=['Fold','ILA','PLA','Error Rate','Precision (Macro)',
    'Precision (Micro)','Recall (Macro)','Recall (Micro)','F1-Score (Macro)','F1-Score (Micro)','AUC']

    results = pd.DataFrame(columns=columns)
    metrics.insert(0,folds)

    for c,m in zip(columns,metrics):
        print(c,m)
        results[c]=m
    
    
    metrics.remove(metrics[0])
    columns.remove(columns[0])
    t1 = "../results/classification_metrics_"+tc+".csv"
    t2 = "../results/avg_metric_"+tc+".txt"
    print("*****************************************************************")
    results.to_csv(t1,index=None)
    f = open(t2,'w')
    for c,m in zip  (columns,metrics):
        avg  = sum(m)/len(m)
        print("AVG. %s = %f" %(c,avg))
        f.write("AVG. %s = %f \n" %(c,avg))
    f.close()
    



def run_folds():
    tc = parse_argument()
    print("Using %s Classifier" %(tc))
    ila_values = []
    pla_values = []
    ee_values = []
    macro_precision_values = []
    micro_precision_values = []
    macro_recall_values = []
    micro_recall_values = []
    macro_f1_values = []
    micro_f1_values = []
    auc_values = []
    for f in range(1,6):
        print("**********************************")
        print("Fold %d" %(f))
        train_path = "../embeddings/train/train_emb_f"+str(f)+".npz"
        test_path = "../embeddings/test/test_emb_f"+str(f)+".npz"
        test_paths_p = "../embeddings/test/test_images_path_f"+str(f)
        x_train,y_train,x_test,y_test,test_paths = upload_data(train_path,test_path,test_paths_p)
        y_pred = classifier(x_train,y_train,x_test,tc)
        ila,pla,error_rate,macro_precision,micro_precision,macro_recall,micro_recall,f1_macro,f1_micro,auc_score =classification_metrics(y_test,y_pred,test_paths)
        ila_values.append(ila)
        pla_values.append(pla)
        ee_values.append(error_rate)
        macro_precision_values.append(macro_precision)
        micro_precision_values.append(micro_precision)
        macro_recall_values.append(macro_recall)
        micro_recall_values.append(micro_recall)
        macro_f1_values.append(f1_macro)
        micro_f1_values.append(f1_micro)
        auc_values.append(auc_score)
        print("**********************************")
    metrics = [ila_values,pla_values,ee_values,macro_precision_values,micro_precision_values,macro_recall_values,micro_recall_values,macro_f1_values,micro_f1_values,auc_values]
    print(len(metrics))
    save_metrics(metrics,tc)
    


if __name__ == '__main__':
    run_folds()
