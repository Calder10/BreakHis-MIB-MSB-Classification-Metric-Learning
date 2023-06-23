import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score, confusion_matrix,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import pstdev
import warnings
from pprint import pprint
warnings.filterwarnings("ignore")


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
    nk = {'40X':19,'100X':3,'200X':5,'400X':3}
    if ct =="SVM":
        clf = SVC()
    else:
        clf = KNeighborsClassifier(nk[m])
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    proba=clf.predict_proba(x_test)
    return y_pred,proba

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
        #print("Patient Code: %s  Patient score: %f" %(p,acc))
    
        
    pla=sum(p_i)/len(patients)
    return pla

def ila_pla(y_test,y_pred,test_paths,proba):
    ila = accuracy_score(y_test,y_pred)
    print("ILA=",ila)
    pla = compute_pla(y_test,y_pred,test_paths)
    print("PLA=",pla)

    prec=precision_score(y_test,y_pred)
    print("Precision=",prec)

    rec=recall_score(y_test,y_pred)
    print("Recall=",rec)

    f1score=f1_score(y_test,y_pred)
    print("F1-Score=",f1score)

    auc=roc_auc_score(y_test,proba[:,1])
    print("AUC=",auc)

    return ila,pla,prec,rec,f1score,auc


def plot_cm(cm,c,mf,f=None):
    if f != None:
        title = "MSB Classificaation - Classifier: %s \n Fold %s Magnification fator %s" %(c,f,mf)
        path = "../plots/confusion matrices/cm_f%s_%s_%s.png" %(f,mf,c)
    else:
        title = "MSB Classificaation - Classifier: %s \n Average Confusion Matrix \n Magnification Factor %s" %(c,mf)
        path = "../plots/confusion matrices/avg_cm_%s_%s.png" %(c,mf)
    labels = ['Benign','Malignant']
    plt.figure(figsize=(4,4))
    plt.suptitle(title)
    ax=sns.heatmap(np.array(cm), annot=True,fmt='g',cmap='Blues',cbar=False,annot_kws={"size": 14})
    ax.set_xticklabels(labels,rotation=45,fontsize=12)
    ax.set_yticklabels(labels,rotation=45,fontsize=12)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.tight_layout()
    plt.savefig(path,format='png')

def avg_cm(cms,tc,mf):
    avg_cm = sum(cms)
    avg_cm = np.rint(avg_cm/5)
    plot_cm(avg_cm,tc,mf,None)

def plot_average_roc_curve(ct,tests, preds,mf):
    title = "MSB Classification - Classifier %s \n Magnification factor %s  Average Roc Curve" %(ct,mf)
    path = "../plots/roc curves/avg_roc_curve_%s_%s.png" %(ct,mf)
    plt.figure()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 1

    for x,y in zip(tests,preds):
        fpr, tpr, t = roc_curve(x,y)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1
    
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.legend(loc="lower right")
    plt.suptitle(title,fontsize=12)
    plt.savefig(path,format = 'png')

def save_metrics(ila,pla,prec,rec,f1,auc,ct):
    folds = range(1,6)

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
    avg_rec=["Avg. Recall"]
    avg_prec=["Avg. Precision"]
    avg_f1=["Avg. F1-Score"]
    avg_auc=["Avg. AUC"]
    print("ILA VALUES:")
    print(ila_df,"\n \n")
    print("PLA VALUES:")
    print(pla_df,"\n \n")


    for key in zip(ila,pla,prec,rec,f1,auc):
        a_ila = sum(ila[key[0]])/len(ila[key[0]])
        sd = pstdev(ila[key[0]])
        avg_ila.append([a_ila,sd])
        a_pla = sum(pla[key[0]])/len(pla[key[0]])
        sd = pstdev(pla[key[0]])
        avg_pla.append([a_pla,sd])

        a_rec = sum(rec[key[0]])/len(rec[key[0]])
        sd = pstdev(rec[key[0]])
        avg_rec.append([a_rec,sd])

        a_prec = sum(prec[key[0]])/len(prec[key[0]])
        sd = pstdev(prec[key[0]])
        avg_prec.append([a_prec,sd])


        a_f1 = sum(f1[key[0]])/len(f1[key[0]])
        sd = pstdev(f1[key[0]])
        avg_f1.append([a_f1,sd])


        a_auc = sum(auc[key[0]])/len(auc[key[0]])
        sd = pstdev(auc[key[0]])
        avg_auc.append([a_auc,sd])
        
        

    t1 = "../results/ila_values_"+ct+".csv"
    t2 = "../results/pla_values_"+ct+".csv"
    t3 = "../results/ila_pla_avg_"+ct+".csv"
    
    avg_df = pd.DataFrame([avg_ila,avg_pla,avg_prec,avg_rec,avg_f1,avg_auc],columns=['Metric','40X','100X','200X','400X'])
    print("*****************************************************************")
    print("AVG ILA and PLA values:")
    print(avg_df.T)
    print("*****************************************************************")


    ila_df.to_csv(t1,index=None)
    pla_df.to_csv(t2,index=None)
    avg_df.to_csv(t3,index=False)

def run_5_folds():
    ct = parse_argument()
    print("Using %s classifier" %(ct))
    magn = ['40X','100X','200X','400X']
    ila_values = {}
    pla_values = {}
    prec_values = {}
    rec_values ={}
    f1_values={}
    auc_values={}

    for m in magn:
        aus_ila=[]
        aus_pla=[]
        aus_prec=[]
        aus_rec=[]
        aus_f1=[]
        aus_auc=[]
        cms = []
        y_tests=[]
        preds=[]
        for f in range(1,6):
            print("**********************************")
            print("Magnification Factor %s -  Fold %d" %(m,f))
            train_path = "../embeddings/"+str(m)+"/train/train_emb_f"+str(f)+".npz"
            test_path = "../embeddings/"+str(m)+"/test/test_emb_f"+str(f)+".npz"
            test_paths_p = "../embeddings/"+str(m)+"/test/test_images_path_f"+str(f)
            x_train,y_train,x_test,y_test,test_paths = upload_data(train_path,test_path,test_paths_p)
            y_pred,proba = classifier(x_train,y_train,x_test,ct,m)
            ila,pla,prec,rec,f1,auc =ila_pla(y_test,y_pred,test_paths,proba)
            aus_ila.append(ila)
            aus_pla.append(pla)
            aus_prec.append(prec)
            aus_rec.append(rec)
            aus_f1.append(f1)
            aus_auc.append(auc)
            cm = confusion_matrix(y_test,y_pred)
            plot_cm(cm,ct,m,f)
            cms.append(cm)
            y_tests.append(y_test)
            preds.append(y_pred)
        print("**********************************")
        ila_values[m]=aus_ila
        pla_values[m]=aus_pla
        prec_values[m]=aus_prec
        rec_values[m]=aus_rec
        f1_values[m]=aus_f1
        auc_values[m]=aus_auc
        avg_cm(cms,ct,m)
        plot_average_roc_curve(ct,y_tests, preds,m)
    save_metrics(ila_values,pla_values,prec_values,rec_values,f1_values,auc_values,ct)

if __name__ == '__main__':
    run_5_folds()
