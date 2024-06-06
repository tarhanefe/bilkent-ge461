import numpy as np 
import matplotlib.pyplot as plt

class Data() :
    def __init__(self):
        self.X_train,self.X_test,self.y_train,self.y_test = self.load_data()
    
    def load_data(self):
        
        train_data = []
        train_labels = []
        with open("train1.txt", 'r') as file:
            for line in file:
                parts = line.strip().split()  
                train_data.append(float(parts[0]))  
                train_labels.append(float(parts[1]))
            X_train,y_train = np.array(train_data), np.array(train_labels)
             
        test_data = []
        test_labels = []
        with open("test1.txt", 'r') as file:
            for line in file:
                parts = line.strip().split()  
                test_data.append(float(parts[0]))  
                test_labels.append(float(parts[1]))
            X_test,y_test = np.array(test_data), np.array(test_labels)
        
        return X_train,X_test,y_train,y_test
    
    def normalize(self):
        self.mean_X_train = np.mean(self.X_train)
        self.mean_X_test = np.mean(self.X_test)
        self.mean_y_train = np.mean(self.y_train)
        self.mean_y_test = np.mean(self.y_test)
        
        self.std_X_train = np.std(self.X_train)
        self.std_X_test = np.std(self.X_test)
        self.std_y_train = np.std(self.y_train)
        self.std_y_test = np.std(self.y_test)
        
        self.X_train_n = (self.X_train-self.mean_X_train)/self.std_X_train
        self.X_test_n = (self.X_test-self.mean_X_test)/self.std_X_test
        self.y_train_n = (self.y_train-self.mean_y_train)/self.std_y_train
        self.y_test_n = (self.y_test-self.mean_y_train)/self.std_y_train
        
    def transform(self,data,stat):
        if stat == 'xtr':
            data = data*self.std_X_train+self.mean_X_train
        elif stat == 'xts':
            data = data*self.std_X_test+self.mean_X_test
        elif stat == 'ytr':
            data = data*self.std_y_train+self.mean_y_train
        elif stat == 'yts':
            data = data*self.std_y_train+self.mean_y_train
        return data

#%% Defining an ANN class that can create models with a hidden layer and no hidden layer 

class ANN():
    
    def __init__(self,hidden_available,number_of_hidden,initalization):
        self.hidden_available = hidden_available
        self.number_of_hidden = number_of_hidden
        self.initialization = initalization
        self.sigmoid = lambda x : 1/(1 + np.exp(-x))
        self.der_sigmoid = lambda x : 1/(1 + np.exp(-x))*(1-1/(1 + np.exp(-x)))
        self.loss = lambda pred,true : (pred-true)**2
        self.der_loss = lambda pred,true : 2*(pred-true)
        
    def initialize_parameters(self,seed):
        if self.initialization == 'type1':
            np.random.seed(seed)
            if not self.hidden_available:
                self.w = np.random.rand()*2-1
                self.b = 0
                model = {'weight': self.w, 'bias': self.b}
                return model
            else:
                self.w0 = (np.random.rand(self.number_of_hidden,1) * 2-1)*np.sqrt(6/self.number_of_hidden+1)
                self.b0 = np.zeros((self.number_of_hidden,1))
                
                self.w1 = (np.random.rand(1,self.number_of_hidden) * 2-1)*np.sqrt(6/self.number_of_hidden+1)
                self.b1 = 0
                model = {'weight0': self.w0, 'bias0': self.b0,'weight1': self.w1, 'bias1': self.b1}
                return model
            
        elif self.initialization == 'type2':
            np.random.seed(seed)
            if not self.hidden_available:
                self.w = np.random.rand()*2-1
                self.b = np.random.rand()*2-1
                model = {'weight': self.w, 'bias': self.b}
                return model
            else:
                self.w0 = (np.random.rand(self.number_of_hidden,1) * 2-1)
                self.b0 = (np.random.rand(self.number_of_hidden,1)*2-1)
                
                self.w1 = (np.random.rand(1,self.number_of_hidden) * 2-1)
                self.b1 = np.random.rand()*2-1
                model = {'weight0': self.w0, 'bias0': self.b0,'weight1': self.w1, 'bias1': self.b1}
                return model
            
        
        elif self.initialization == 'type3':
            np.random.seed(seed)
            if not self.hidden_available:
                self.w = 0
                self.b = 0
                model = {'weight': self.w, 'bias': self.b}
                return model
            else:
                self.w0 = np.zeros((self.number_of_hidden,1))
                self.b0 = np.zeros((self.number_of_hidden,1))
                
                self.w1 = np.zeros((1,self.number_of_hidden))
                self.b1 = 0
                model = {'weight0': self.w0, 'bias0': self.b0,'weight1': self.w1, 'bias1': self.b1}
                return model
    
    def forward(self,x):
        if self.hidden_available:
            hidden  = self.w0 * x + self.b0
            self.out_hidden = self.sigmoid(hidden)
            self.pred = self.w1 @ self.out_hidden + self.b1
        else:
            self.pred = self.w * x + self.b 
        
    def calc_grad(self, x, y):
        self.forward(x)
        
        if self.hidden_available:
            sigma = self.der_loss(self.pred, y)
            h = self.out_hidden
            self.gradw1 = np.dot(sigma, h.T)
            self.gradb1 = sigma
            sigma_hidden = np.dot(self.w1.T, sigma) * self.der_sigmoid(h)
            x = np.atleast_2d(x).T  
            self.gradw0 = np.dot(sigma_hidden, x.T)
            self.gradb0 = sigma_hidden
        else:
            self.gradw = self.der_loss(self.pred, y) * x
            self.gradb = self.der_loss(self.pred, y)

            
    def update(self,lr):
        if self.hidden_available:
            self.w0 = self.w0 - lr*self.gradw0 
            self.w1 = self.w1 - lr*self.gradw1
            self.b0 = self.b0 - lr*self.gradb0 
            self.b1 = self.b1 - lr*self.gradb1 
        else:
            self.w = self.w - lr*self.gradw
            self.b = self.b - lr*self.gradb

    def train(self,X,y,lr,epoch,seed):
        self.initialize_parameters(seed)
        np.random.seed(seed)
        self.train_losses = []
        for ep in range(epoch):
            print("===================")
            print("Epoch {} of {}: ".format(ep,epoch-1))
            indices = np.random.permutation(len(X))
            # Apply the permutation to both arrays
            ep_X = X[indices]
            ep_y = y[indices]
            for i in range(len(ep_X)):
                self.calc_grad(ep_X[i], ep_y[i])
                self.update(lr)
                
            loss = 0 
            for j in range(len(ep_X)):
                self.forward(ep_X[j])
                y_pred = self.pred
                loss += self.loss(y_pred,ep_y[j])
            self.train_losses.append(loss.item())
            print("Loss: {}".format(loss))