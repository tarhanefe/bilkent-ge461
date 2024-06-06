#%% Importing necessary libraries
import os 
import numpy as np 
import matplotlib.pyplot as plt
from include import *
os.chdir("/Users/efetarhan/Desktop/GE461 HW3");

#%% Data Visualization 
dataset = Data()
dataset.normalize()

X_train,X_test,y_train,y_test = dataset.X_train,dataset.X_test,dataset.y_train,dataset.y_test
X_train_n,X_test_n,y_train_n,y_test_n = dataset.X_train_n,dataset.X_test_n,dataset.y_train_n,dataset.y_test_n

plt.figure(1,dpi = 600)
plt.plot(X_train,y_train,'bo',markersize = 3)
plt.grid(True)
plt.plot(X_test,y_test,'ro',markersize = 3)
plt.grid(True)
plt.legend(["Train Data","Test Data"])

#%% ANN - Linear Regression Comparison
np.random.seed(42)
NN1 = ANN(False,3,'type1')
NN2 = ANN(True,3,'type1')

lr = 0.025
epoch = 19743

NN1.train(X_train_n, y_train_n, lr, epoch,42)  
NN2.train(X_train_n, y_train_n, lr, epoch,42)
xrange = np.arange(np.min(X_train_n),np.max(X_train_n),0.001)
NN1.forward(xrange)
NN2.forward(xrange)
test_pred1 = dataset.transform(NN1.pred,'ytr')
test_pred2 = dataset.transform(NN2.pred,'ytr')
plt.figure(dpi = 600)
plt.plot(dataset.transform(xrange,'xtr'),test_pred1,'m-')
plt.plot(dataset.transform(xrange,'xtr'),test_pred2[0],'r-')
plt.plot(X_train,y_train,'bo',markersize = 3)
plt.grid(True)
plt.legend(['No Hidden Layer','Single Hidden Layer','True Labels'])

#%% Hidden Unit Comparsion
NN1 = ANN(True,1,'type1')
lr = 0.01
epoch = 2321
NN1.train(X_train, y_train, lr, epoch,42)  
NN1.forward(np.arange(-1.5,2,0.001))

NN2 = ANN(True,2,'type1')
lr = 0.015
epoch = 3148
NN2.train(X_train, y_train, lr, epoch,42)  
NN2.forward(np.arange(-1.5,2,0.001))

NN3 = ANN(True,3,'type1')
lr = 0.03
epoch = 9995
NN3.train(X_train, y_train, lr, epoch,42)  
NN3.forward(np.arange(-1.5,2,0.001))

NN4 = ANN(True,4,'type1')
lr = 0.03
epoch = 9936
NN4.train(X_train, y_train, lr, epoch,42)  
NN4.forward(np.arange(-1.5,2,0.001))

fig, axs = plt.subplots(2, 2, dpi=600, figsize=(10, 10))
axs[0,0].plot(np.arange(-1.5,2,0.001),NN1.pred[0],'r-')
axs[0,0].plot(X_train,y_train,'bo',markersize = 3)
axs[0,0].set_title("# Hidden Units = 1")
axs[0,0].grid(True)

axs[0,1].plot(np.arange(-1.5,2,0.001),NN2.pred[0],'r-')
axs[0,1].plot(X_train,y_train,'bo',markersize = 3)
axs[0,1].set_title("# Hidden Units = 2")
axs[0,1].grid(True)

axs[1,0].plot(np.arange(-1.5,2,0.001),NN3.pred[0],'r-')
axs[1,0].plot(X_train,y_train,'bo',markersize = 3)
axs[1,0].set_title("# Hidden Units = 3")
axs[1,0].grid(True)

axs[1,1].plot(np.arange(-1.5,2,0.001),NN4.pred[0],'r-')
axs[1,1].plot(X_train,y_train,'bo',markersize = 3)
axs[1,1].set_title("# Hidden Units = 4")
axs[1,1].grid(True)

#%% Normalization Comparison
NN1 = ANN(True,3,'type1')
NN2 = ANN(True,3,'type1')

lr = 0.03
epoch = 20000
NN1.train(X_train_n, y_train_n, lr, epoch,42)

lr = 0.03
epoch = 20000
NN2.train(X_train, y_train, lr, epoch,42)

plt.figure(dpi = 600,figsize = (8,8))
xrange_n = np.arange(np.min(X_train_n),np.max(X_train_n),0.001)
NN1.forward(xrange_n)
NN2.forward(dataset.transform(xrange_n,'xtr'))
plt.plot(X_train,y_train,'bo')
plt.plot(dataset.transform(xrange_n,'xtr'),dataset.transform(NN1.pred[0],'ytr'),'r-')
plt.plot(dataset.transform(xrange_n,'xtr'),NN2.pred[0],'m-')
plt.grid(True)
plt.xlabel('Feature')
plt.ylabel('Label')
plt.legend(['Train Data','With Normalization','Without Normalization'])

#%% Learning Rate Comparison
NN = ANN(True,3,'type1')
lr = [0.01,0.03,0.05]
epoch = 20000
errors = []

for i in lr:
    NN.train(X_train_n, y_train_n, i, epoch,42)
    errors.append(NN.train_losses)
    
fig, axs = plt.subplots(1, 3, dpi=600, figsize=(20, 7))
axs[0].plot(errors[0],'r-')
axs[0].set_title("Training for each epoch w/ Lr = 0.01")
axs[0].grid(True)
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("MSE")

axs[1].plot(errors[1],'r-')
axs[1].set_title("Training for each epoch w/ Lr = 0.03")
axs[1].grid(True)
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("MSE")

axs[2].plot(errors[2],'r-')
axs[2].set_title("Training for each epoch w/ Lr = 0.05")
axs[2].grid(True)
axs[2].set_xlabel("Epochs")
axs[2].set_ylabel("MSE")

#%% Initialization Comparison
NN1 = ANN(True,3,'type1')
NN2 = ANN(True,3,'type2')
NN3 = ANN(True,3,'type3')
NN_s = [NN1,NN2,NN3]
lr = 0.03
epoch = 20000

for i in NN_s:
    i.train(X_train_n, y_train_n, lr, epoch,42)
    errors.append(i.train_losses)

fig, axs = plt.subplots(1, 3, dpi=600, figsize=(20, 7))
axs[0].plot(errors[-1],'m-')
axs[0].set_title("Type 1 Weight Initialization w/ Lr = 0.03")
axs[0].grid(True)
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("MSE")

axs[1].plot(errors[-2],'m-')
axs[1].set_title("Type 2 Weight Initialization w/ Lr = 0.03")
axs[1].grid(True)
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("MSE")

axs[2].plot(errors[-3],'m-')
axs[2].set_title("Type 3 Weight Initialization w/ Lr = 0.03")
axs[2].grid(True)
axs[2].set_xlabel("Epochs")
axs[2].set_ylabel("MSE")

#%% Best Configured Model
NN = ANN(True,3,'type1')
lr = 0.03
epoch = 19028
xrange = np.arange(np.min(X_train_n),np.max(X_train_n),0.001)
NN.train(X_train_n, y_train_n, lr, epoch,42)    
NN.forward(xrange)
test_pred = NN.pred
plt.figure(dpi = 600)
plt.plot(dataset.transform(xrange,'xtr'),dataset.transform(test_pred[0],'ytr'),'r-')
plt.plot(X_test,y_test,'bo',markersize = 3)
plt.grid(True)
plt.legend(['Predicted Function','True Labels'])

plt.figure(dpi =  600)
plt.plot(NN.train_losses,'-mo',markersize = 3)
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error Loss")
plt.grid(True)

NN.forward(X_train_n)
train_loss = np.mean((dataset.transform(NN.pred,'ytr')-y_train)**2)
NN.forward(X_test_n)
test_loss = np.mean((dataset.transform(NN.pred,'yts')-y_test)**2)

print(test_loss)

#%% Hidden Unit Part C
training_conf = [(False,0),(True,2),(True,4),(True,8),(True,16),(True,32)]
lr_list = [0.01,0.01,0.03,0.03,0.025,0.025]
epoch_list = [17007,2750,48586,17106,32272,29526]
NN_s = []

for trial_cnt in range(len(training_conf)):
    lr = lr_list[trial_cnt]
    hidden_bool = training_conf[trial_cnt][0]
    layer_cnt = training_conf[trial_cnt][1]
    epoch = epoch_list[trial_cnt]
    NN = ANN(hidden_bool,layer_cnt,'type1')
    NN.train(X_train_n, y_train_n, lr, epoch,42)  
    NN_s.append(NN)
    

for i in range(len(NN_s)): 
    NN = NN_s[i]
    range_x = np.arange(np.min(X_train_n),np.max(X_train_n),0.001)
    NN.forward(range_x)
    plt.figure(dpi = 600)
    plt.plot(X_train,y_train,'bo')
    if i !=  0:
        plt.plot(dataset.transform(range_x,'xtr'),dataset.transform(NN.pred[0],'ytr'),'r-')
    else:
        plt.plot(dataset.transform(range_x,'xtr'),dataset.transform(NN.pred,'ytr'),'r-')
    plt.xlabel('Feature')
    plt.ylabel('Label')
    plt.grid(True)
    plt.legend(['Training Data','Fitted Function'])

variance_test = []
mean_test = []
for i in range(len(training_conf)):
    NN = NN_s[i]
    NN.forward(X_test_n)
    mean_err = np.mean((y_test-dataset.transform(NN.pred,'yts'))**2) 
    var_err = np.std((y_test-dataset.transform(NN.pred,'yts'))**2)
    mean_test.append(mean_err)
    variance_test.append(var_err)
    
variance_train = []
mean_train = []
for i in range(len(training_conf)):
    NN = NN_s[i]
    NN.forward(X_train_n)
    mean_err = np.mean((y_train-dataset.transform(NN.pred,'ytr'))**2) 
    var_err = np.std((y_train-dataset.transform(NN.pred,'ytr'))**2)
    mean_train.append(mean_err)
    variance_train.append(var_err)
    


