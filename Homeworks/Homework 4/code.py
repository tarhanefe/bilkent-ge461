#%% Importing necessary libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import os 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings("ignore")
#%% Read the data 
df = pd.read_csv('falldetection_dataset.csv', header=None)
data = df.values
data = data[:,1:]
X,y = data[:,1:],data[:,0] 
#%% PART A - PCA 

scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 10),dpi = 600)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Projection of the Data to 2D')
plt.show()
#%% Creating different number of clusters using the dataset

clusters = [KMeans(n_clusters=i, random_state=42).fit_predict(X_pca) for i in range(2,11)]
fig, axes = plt.subplots(3, 3, figsize=(15, 15),dpi = 600) 
for i in range(3):
    for j in range(3):
        axes[i, j].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters[i*3+j], cmap='Accent')  
        axes[i, j].set_title(f'{i*3+j+2}-means Clustering')
        axes[i, j].set_xlabel('Feature 1')
        axes[i, j].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

#%% Result of the k means on PCA 
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)
plt.figure(dpi = 600, figsize = (10,10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='Accent')  
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')  
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


#%% Acutal data with PCA 
y_bin = y == 'F'

plt.figure(dpi=600, figsize=(10, 10))

plt.scatter(X_pca[y_bin, 0], X_pca[y_bin, 1], color='blue', label='F')
plt.scatter(X_pca[~y_bin, 0], X_pca[~y_bin, 1], color='orange', label='NF')

plt.title('Binary Classification Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.legend(title="Classification")

plt.show()
#%% Overlapping with acutal 

s11 = np.sum((y =='NF') & (clusters == 0))
s12 = np.sum((y =='NF') & (clusters == 1))
s21 = np.sum((y =='F') & (clusters == 0))
s22 = np.sum((y =='F') & (clusters == 1))

plt.figure(figsize=(5, 5),dpi = 600) 
sns.heatmap([[s11,s12],[s21,s22]], annot=True, cmap='coolwarm',fmt=".1f" )  
plt.show()
acc = (s11+s22)/(s11+s12+s21+s22)
recall = s22 / (s22 + s21)
precision = s22 / (s22 + s12)
f1 = 2*precision*recall / (precision + recall)
print("===========================")
print("Classification Scores: ")
print("---------------------------")
print("Accuracy: {}".format(acc))
print("Recall: {}".format(recall))
print("Precision: {}".format(precision))
print("F1 Score: {}".format(f1))
print("===========================")

#%% PART B: SVM

# Split data into train+validation (85%) and test (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Parameters for cross-validation
n_splits = 20
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize the SVM classifier parameters
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
lr = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
accuracy_dict = {k: np.zeros(len(lr)) for k in kernel}

# Cross-validation to find the best parameters
for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    for k in kernel:
        for idx, C in enumerate(lr):
            svm = SVC(C=C, kernel=k)
            svm.fit(X_train, y_train)
            y_val_pred = svm.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            accuracy_dict[k][idx] += acc

# Average the accuracies over the folds
for k in kernel:
    accuracy_dict[k] /= n_splits

plt.figure(dpi=600, figsize=(9, 6))
for k in kernel:
    plt.semilogx(lr, accuracy_dict[k], label=k.capitalize())
plt.grid(True)
plt.xlabel("Regularization Parameter C")
plt.ylabel("Average Validation Accuracy")
plt.legend()
plt.show()

best_kernel, best_c_idx = max(((k, np.argmax(v)) for k, v in accuracy_dict.items()), key=lambda x: accuracy_dict[x[0]][x[1]])
best_c = lr[best_c_idx]

best_svm = SVC(C=best_c, kernel=best_kernel)
best_svm.fit(X_train_val, y_train_val)
y_test_pred = best_svm.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy with best C={best_c} and kernel={best_kernel}: {test_acc}")

#%% Part B: MLP

# Split data into train+validation (85%) and test (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Parameters for cross-validation
n_splits = 5  
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define hyperparameters to test
learning_rates = [0.00001,0.0001, 0.001, 0.01, 0.1]
optimizers = ['adam', 'sgd']
layer_sizes = [(32,), (64,), (32, 32), (64, 64)]

fixed_epochs = 400 

# Initialize dictionary to store accuracies
accuracy_dict = {(lr, opt, ls): [] for lr in learning_rates for opt in optimizers for ls in layer_sizes}

# Cross-validation to find the best parameters
for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    for lr in learning_rates:
        for opt in optimizers:
            for ls in layer_sizes:
                mlp = MLPClassifier(hidden_layer_sizes=ls, learning_rate_init=lr, max_iter=fixed_epochs, solver=opt, random_state=42)
                mlp.fit(X_train, y_train)
                y_val_pred = mlp.predict(X_val)
                acc = accuracy_score(y_val, y_val_pred)
                accuracy_dict[(lr, opt, ls)].append(acc)

# Average the accuracies over the folds
for key in accuracy_dict.keys():
    accuracy_dict[key] = np.mean(accuracy_dict[key])

# Plot Learning Rate vs. Average Validation Accuracy
plt.figure(dpi=600, figsize=(12, 8))
for opt in optimizers:
    for ls in layer_sizes:
        accuracies = [accuracy_dict[(lr, opt, ls)] for lr in learning_rates]
        plt.semilogx(learning_rates, accuracies, marker='o', label=f"{opt.upper()}, Layers: {ls}")

plt.title("Validation Accuracy by Learning Rate, Optimizer, and Layer Configuration")
plt.xlabel("Learning Rate")
plt.ylabel("Average Validation Accuracy")
plt.legend(title="Optimizer and Layer Configurations", loc='best')
plt.grid(True)
plt.show()

best_params = max(accuracy_dict, key=accuracy_dict.get)
best_lr, best_optimizer, best_layers = best_params

best_mlp = MLPClassifier(hidden_layer_sizes=best_layers, learning_rate_init=best_lr, max_iter=fixed_epochs, solver=best_optimizer, random_state=42, verbose=True)
best_mlp.fit(X_train_val, y_train_val)

training_losses = best_mlp.loss_curve_
plt.figure(dpi=600, figsize=(12, 8))
plt.plot(training_losses, marker='o', linestyle='-')
plt.title("Training Error per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Training Loss (Log Loss)")
plt.grid(True)
plt.show()

y_test_pred = best_mlp.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy with best params - LR: {best_lr}, Optimizer: {best_optimizer}, Layers: {best_layers}: {test_acc}")
