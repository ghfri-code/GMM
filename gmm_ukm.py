# Import Modules:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM

#-------------------define functions--------------------------------

# Plot Functions:

def plot_data_2d(X, Y, d1_idx, d2_idx, colors, title):
    n_classes = len(np.unique(Y))
    fig, ax = plt.subplots()   
    for i in range(n_classes):           
        plt.scatter(X[:, d1_idx][np.where(Y == i)[0]],
                    X[:, d2_idx][np.where(Y == i)[0]],
                    c=colors[i]) 
    
    ax.set_ylabel('Feature 1')
    ax.set_xlabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    plt.show()

# GridSearch Functions:

def MyGridSearchCV_GMM(X, Y, n_components_range, folds, times):
    
    results = {}
    mean_train_accuracies_k_time = []
    mean_test_accuracies_k_time = []

    for n, k in enumerate(n_components_range):
        gmm_model = GMM(n_components=k)
    
        mean_train_accuracies_k_fold = []
        mean_test_accuracies_k_fold = []
        std_train_accuracies_k_fold = []
        std_test_accuracies_k_fold = []
        for i in range(times):
            kf = KFold(n_splits=folds, random_state=None, shuffle=True)        
            train_accuracies_per_fold = []
            test_accuracies_per_fold = []
            for x, y in kf.split(range(np.shape(X)[0])):
                x_train, y_train = X[x], Y[x]
                x_test, y_test = X[y], Y[y]
        
                models = []
                Pb_train = np.zeros((np.shape(x_train)[0], n_classes))
                Pb_test = np.zeros((np.shape(x_test)[0], n_classes))
                for i in range(n_classes):           
                    models.append(gmm_model.fit(x_train[np.where(y_train == i)[0]]))    
                    Pb_train[:, i] = models[i].score_samples(x_train)
                    Pb_test[:, i] = models[i].score_samples(x_test)
    
                y_pred_train = np.argmax(Pb_train, axis=1) 
                y_pred_test = np.argmax(Pb_test, axis=1) 
                train_accuracies_per_fold.append(accuracy_score(y_train, y_pred_train))  
                test_accuracies_per_fold.append(accuracy_score(y_test, y_pred_test))  
    
            mean_train_accuracies_k_fold.append(np.mean(train_accuracies_per_fold)) 
            mean_test_accuracies_k_fold.append(np.mean(test_accuracies_per_fold))
            std_train_accuracies_k_fold.append(np.std(train_accuracies_per_fold))    
            std_test_accuracies_k_fold.append(np.std(test_accuracies_per_fold))    
    
        mean_train_accuracies_k_time.append(np.max(mean_train_accuracies_k_fold))
        mean_test_accuracies_k_time.append(np.max(mean_test_accuracies_k_fold))

        
        results[str(k)]={'train_acc': round(mean_train_accuracies_k_time[n], 2),
                         'test_acc': round(mean_test_accuracies_k_time[n], 2)}
    
    best_k = n_components_range[np.argmax(mean_test_accuracies_k_time)]
    
    return results, best_k, GMM(n_components=best_k)


# Load & Prepare Dataset:     
def load_dataset(path): 
    df = pd.read_excel(path, sheet_name=['Training_Data', 'Test_Data'])
    df1 = df['Training_Data']
    df2 = df['Test_Data']
    X_train = np.array(df1.drop(df1.columns[5], axis=1)) 
    Y_train = np.array(df1.drop(df1.columns[[0,1,2,3,4]], axis=1))[:, 0]
    X_test = np.array(df2.drop(df2.columns[5], axis=1)) 
    Y_test = np.array(df2.drop(df2.columns[[0,1,2,3,4]], axis=1))[:, 0]
    encoder = LabelBinarizer()
    Y_train = np.argmax(encoder.fit_transform(Y_train), axis=1)
    Y_test = np.argmax(encoder.fit_transform(Y_test), axis=1)
    return X_train, X_test, Y_train, Y_test
        
#-------------------call functions--------------------------------    
# Configure Parameters:    
data_path = "UKM.xls"
colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
covariance_type_range = ['full']# ['spherical', 'tied', 'diag', 'full']
n_components_range = [1,5,10]
param_grid = dict(n_components=n_components_range, covariance_type=covariance_type_range)

# Load & Plot Dataset: 
X_train, X_test, Y_train, Y_test = load_dataset(data_path)

X = np.concatenate([X_train, X_test], axis=0)
Y = np.concatenate([Y_train, Y_test], axis=0)

n_classes = len(np.unique(Y))

s1, s2 = 0, 1   # selected features to plot
plot_data_2d(X_train, Y_train, s1, s2, colors, 'plot-2d (train data)')
plot_data_2d(X_test, Y_test, s1, s2, colors, 'plot-2d (test data)')

for k in n_components_range:
    gmm_model = GMM(n_components=k)
    models = []
    Pb_test = np.zeros((np.shape(X_test)[0], n_classes))
    for i in range(n_classes):           
        models.append(gmm_model.fit(X_train[np.where(Y_train == i)[0]]))    
        Pb_test[:, i] = models[i].score_samples(X_test)
    Y_pred_test = np.argmax(Pb_test, axis=1) 
    plot_data_2d(X_test, Y_pred_test, s1, s2, colors, 'plot-2d (predict test data & k = ' + str(k) + ')')
    

# Find Best K for GMM (k-time-k-fold cv):
times = 5
folds = 5
results, best_k, best_gmm = MyGridSearchCV_GMM(X, Y, n_components_range, folds, times)
print("The best n component is: ", best_k)
print(results)
