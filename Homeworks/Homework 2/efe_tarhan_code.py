#%% Importing Necessary Libraries
import numpy as np 
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split


new_directory = "/Users/efetarhan/Desktop/GE461 HW2"
os.chdir(new_directory)

#%% Read Data Features

X_list = []
with open('fashion_mnist_data.txt', 'r') as file:
    for line in file:
        line =[float(i) for i in line.split(" ")]
        line = np.array(line)
        X_list.append(line)
        
X_list = np.array(X_list)

#%% Read Data Labels

y_list = []
with open('fashion_mnist_labels.txt', 'r') as file:
    for line in file:
        line =[float(i) for i in line.split(" ")]
        line = np.array(line)
        y_list.append(line)
        
y_list = np.array(y_list)


#%% Centralize the Dataset

X_centered = X_list - np.mean(X_list,axis = 0)

#%% Train-Test Split the Data

X_train, X_test, y_train, y_test = train_test_split(X_centered, y_list, test_size=0.5, random_state=42)


#%% Apply the PCA operation

def GaussianPred(X_train, X_test, y_train, y_test):
    classes = np.unique(y_train)
    models = {}
    
    for cls in classes:
        indices = np.where(y_train == cls)[0]
        mean = np.mean(X_train[indices], axis=0)
        cov = np.cov(X_train[indices].T)
        models[cls] = multivariate_normal(mean=mean, cov=cov,seed = 42)
    
    def predict(X):
        probs = np.array([models[cls].pdf(X) for cls in classes])
        return classes[np.argmax(probs, axis=0)]
    
    y_pred_train = predict(X_train)
    y_pred_test = predict(X_test)
    
    train_err = np.mean(y_pred_train != y_train.squeeze()) * 100
    test_err = np.mean(y_pred_test != y_test.squeeze()) * 100
    
    
    return train_err, test_err

def applyPCA(n_components,X):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    return X_pca,pca


#%% Plot the Eigenvalues of the PCA 
X_PCA,pca = applyPCA(n_components = None,X = X_train)
eigenvalues = pca.explained_variance_

from itertools import accumulate
a = list(accumulate(eigenvalues))
plt.figure(2,dpi = 600,figsize = (20,10))
plt.plot(np.arange(len(eigenvalues)),eigenvalues,'ro-',ms = 5)
plt.xlabel("Eigenvalues")
plt.ylabel("Explained Variance (Value)")
plt.grid("True")
plt.title("Eigenvalues of the PCA Applied on the Dataset")


#%% Plot the Sample Mean as an Image
a, b, c, d = train_test_split(X_list, y_list, test_size=0.5, random_state=42)
X_mean = np.mean(a, axis=0)
plt.figure(3,dpi = 600)
plt.imshow(X_mean.reshape((28,28)).T,cmap = 'gray')

#%% Plot Eigenvectors as Images
m = 9
fig, axs = plt.subplots(m, m,figsize = (40,40) ,dpi = 300)
eigenvectors = pca.components_
plt.figure(4,dpi = 600)
for i in range(m*m):
    axs[i//m,i%m].imshow(eigenvectors[i,:].reshape((28,28)).T,cmap = 'grey')
    axs[i//m,i%m].set_title("Eigenvector No:" + str(i+1))


#%% Train multivariate gaussian for principle component projection
pca_list = [i for i in range(10,380,5)]
test_err = []
train_err = []
for i in pca_list:
    X_train_n,pca = applyPCA(i,X_train)
    X_test_n = pca.transform(X_test)
    train_acc, test_acc = GaussianPred(X_train_n, X_test_n, y_train, y_test)    
    train_err.append(train_acc)
    test_err.append(test_acc)
    
#%% Plotting training and testing accuracies
plt.figure(5,dpi = 600,figsize = (20,10))
plt.plot(pca_list,train_err,'-bs')
plt.plot(pca_list,test_err,'-r^')
plt.grid(True)
plt.legend(['Train Error', 'Test Error'])
plt.title('Training and Testing Error for Different Number of Principle Components')
plt.xlabel('Number of Components')
plt.ylabel('CLassification Error Percentage (%)')

#%% Train multivariate gaussian for random component projection

def RandomTransform(data,new_feature):
    old_feature = data.shape[1]
    A = np.random.randn(old_feature, new_feature)
    column_norms = np.linalg.norm(A, axis=0, keepdims=True)
    A_normalized = A / column_norms
    return data@A_normalized
    
    
from sklearn.random_projection import GaussianRandomProjection
pca_list = [i for i in range(10,380,5)]
test_err = []
train_err = []
for i in pca_list:
    X_rp = RandomTransform(X_centered,i)
    X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(X_rp, y_list, test_size=0.5, random_state=42)
    train_acc, test_acc = GaussianPred(X_train_R, X_test_R, y_train_R, y_test_R)    
    train_err.append(train_acc)
    test_err.append(test_acc)
    

#%% Plotting training and testing accuracies
plt.figure(6,dpi = 600,figsize = (20,10))
plt.plot(pca_list,train_err,'-bs')
plt.plot(pca_list,test_err,'-r^')
plt.grid(True)
plt.legend(['Train Error', 'Test Error'])
plt.title('Training and Testing Error for Different Number of Random Components')
plt.xlabel('Number of Components')
plt.ylabel('CLassification Error Percentage (%)')

#%% Train multivariate gaussian for random component projection
cnt = 0
pca_list = [i for i in range(10,380,15)]
pca_cnt = len(pca_list)
test_err = []
train_err = []
for i in pca_list:
    n_neighbors = 4  # Number of neighbors to consider for each point.
    isomap = Isomap(n_neighbors=n_neighbors, n_components=i)
    X_rp = isomap.fit_transform(X_centered,y_list)
    X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(X_rp, y_list, test_size=0.5, random_state=42)
    train_acc, test_acc = GaussianPred(X_train_R, X_test_R, y_train_R, y_test_R)    
    train_err.append(train_acc)
    test_err.append(test_acc)
    cnt += 1
    print(str(cnt) + '/' + str(pca_cnt))

#%% Plotting training and testing accuracies
plt.figure(7,dpi = 600,figsize = (20,10))
plt.plot(pca_list,train_err,'-bs')
plt.plot(pca_list,test_err,'-r^')
plt.grid(True)
plt.legend(['Train Error', 'Test Error'])
plt.title('Training and Testing Error for Different Number of Isomap Components')
plt.xlabel('Number of Components')
plt.ylabel('CLassification Error Percentage (%)')

#%% TSNE for dimensionality reduction


tsne = TSNE(n_components=2, random_state=42,perplexity= 30)
X_tsne = tsne.fit_transform(X_list)
#%%
plt.figure(8,dpi = 600,figsize = (20,10))

class_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

y_list = y_list.flatten()
unique_classes = np.unique(y_list)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))

for label in np.unique(y_list):
    idx = y_list == label
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=class_names[label], s=50, alpha=0.6)

plt.legend()
plt.title('t-SNE visualization of the Dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.show()




