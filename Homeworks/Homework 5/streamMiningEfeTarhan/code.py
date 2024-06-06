# Code by Efe Tarhan
# %% Importing necessary libraries
import numpy as np
import pandas as pd
from skmultiflow.data import SEAGenerator, AGRAWALGenerator
from skmultiflow.drift_detection import ADWIN
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import AdaptiveRandomForest
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\efeta\OneDrive\Desktop\GE461 HW5")
#%% Important Function Definitions

#SEA DataStream Generation and CSV file creation function 
def generate_sea_drift(n_samples=100000, drift_points=[25000, 50000, 75000], seed=42, save=False):
    '''
    Generate SEA DataStream with drift points and save as CSV file if specified.
    
    Parameters:
    - n_samples (int): Number of samples to generate.
    - drift_points (list): List of indices where drift should occur.
    - seed (int): Random seed for reproducibility.
    - save (bool): Whether to save the generated data as a CSV file.

    Returns:
    - sea_data (ndarray): Generated SEA data.
    - sea_targets (ndarray): Generated SEA targets.
    '''
    np.random.seed(seed)
    func_list = np.random.permutation([0, 1, 2, 3])
    generator = SEAGenerator(classification_function=func_list[0], random_state=seed)
    data = []
    targets = []
    cntr = 1
    for i in range(n_samples):
        X, y = generator.next_sample()
        if i in drift_points:
            generator = SEAGenerator(classification_function=func_list[cntr], random_state=42)
            cntr += 1
        data.append(X[0])
        targets.append(y[0])
    sea_data = np.array(data)
    sea_targets = np.array(targets)
    if save:
        sea_targets_reshaped = sea_targets.reshape(-1, 1)
        combined_data = np.concatenate([sea_data, sea_targets_reshaped], axis=1)
        column_names = [f'feature_{i+1}' for i in range(sea_data.shape[1])] + ['target']
        df = pd.DataFrame(combined_data, columns=column_names)
        csv_file_path = 'SEAdata.csv'
        df.to_csv(csv_file_path, index=False)
    return sea_data, sea_targets

#AGRAWAL DataStream Generation and CSV file creation function 
def generate_agrawal_drift(n_samples=100000, drift_points=[25000, 50000, 75000], seed=42, save=False):
    '''
    Generate AGRAWAL DataStream with drift points and save as CSV file if specified.
    
    Parameters:
    - n_samples (int): Number of samples to generate.
    - drift_points (list): List of indices where drift should occur.
    - seed (int): Random seed for reproducibility.
    - save (bool): Whether to save the generated data as a CSV file.

    Returns:
    - sea_data (ndarray): Generated SEA data.
    - sea_targets (ndarray): Generated SEA targets.
    '''
    
    np.random.seed(seed)
    func_list = np.random.permutation(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  
    generator = AGRAWALGenerator(
        classification_function=func_list[0], random_state=seed)
    data = []
    targets = []
    cntr = 1
    for i in range(n_samples):
        X, y = generator.next_sample()
        if i in drift_points:
            generator = AGRAWALGenerator(
                classification_function=func_list[cntr], random_state=seed)
            cntr = (cntr + 1) % len(func_list)
        data.append(X[0])
        targets.append(y[0])
    agrawal_data = np.array(data)
    agrawal_targets = np.array(targets)
    if save == True:
        agrawal_targets_reshaped = agrawal_targets.reshape(-1, 1)
        combined_data = np.concatenate(
            [agrawal_data, agrawal_targets_reshaped], axis=1)
        column_names = [
            f'feature_{i+1}' for i in range(agrawal_data.shape[1])] + ['target']
        df = pd.DataFrame(combined_data, columns=column_names)
        csv_file_path = 'AGRAWALdata.csv'
        df.to_csv(csv_file_path, index=False)
    return agrawal_data, agrawal_targets

# Function for loading real datasets: "spam" and "elec"
def load_dataset(data_name):
    '''
    Load dataset from a CSV file.

    Parameters:
    data_name (str): The name of the CSV file (without the extension).

    Returns:
    data_array (numpy.ndarray): The array containing the features of the dataset.
    target_array (numpy.ndarray): The array containing the target values of the dataset.
    '''
    df = pd.read_csv(f"{data_name}.csv")
    features = df.drop('target', axis=1)
    target = df['target']
    data_array = features.values
    target_array = target.values
    return data_array, target_array

#The function to inject adversarial attacks

def inject_adversarial_attacks(data, targets, attack_points=[40000, 60000], flip_percentage=[0.1, 0.2]):
    '''
    Injects adversarial attacks into the targets array at specified attack points.

    Parameters:
    - data: numpy array
        The input data.
    - targets: numpy array
        The target labels.
    - attack_points: list, optional
        The indices where the attacks will be injected. Default is [40000, 60000].
    - flip_percentage: list, optional
        The percentage of labels to flip at each attack point. Default is [0.1, 0.2].

    Returns:
    - data: numpy array
        The modified input data.
    - targets: numpy array
        The modified target labels.
    '''
    for point, flip in zip(attack_points, flip_percentage):
        start = point
        end = point + 500
        num_flips = int((end - start) * flip)
        flip_indices = np.random.choice(
            range(start, end), num_flips, replace=False)
        targets[flip_indices] = 1 - targets[flip_indices] 
    return data, targets
# %% A Dataset Generator Object that combines the previous functions
class Dataset():
    '''
    Class representing a dataset.
    '''
    def __init__(self, dataset, adversarial_attack=False, save=False):
        self.dataset = dataset
        self.adversarial_attack = adversarial_attack
        self.save = save
        self.generate()
        self.inject_attack()
        
    def generate(self):
        '''
        Generates the data and labels for the dataset based on the specified dataset type.
        '''
        if self.dataset == "SEA":
            self.data, self.label = generate_sea_drift(n_samples=100000, drift_points=[25000, 50000, 75000], seed=42, save=self.save)
        elif self.dataset == "AGRAWAL":
            self.data, self.label = generate_agrawal_drift(n_samples=100000, drift_points=[25000, 50000, 75000], seed=42, save=self.save)
        elif self.dataset == "spam":
            self.data, self.label = load_dataset(self.dataset)
        elif self.dataset == "elec":
            self.data, self.label = load_dataset(self.dataset)
     
    def inject_attack(self):
        '''
        Injects adversarial attacks into the data and labels if adversarial_attack is True.
        '''
        if self.adversarial_attack:
            self.data, self.label = inject_adversarial_attacks(self.data, self.label, attack_points=[40000, 60000], flip_percentage=[0.1, 0.2])
#%% Datastream class is responsible for evaluating the performance of the classifiers
class DataStream():
    '''
    A class representing a data stream for evaluating classifiers.

    Parameters:
    - task_type (str): The type of task, either "classical" or "adversarial".
    - dataset (str): The dataset to use for evaluation.
    - adversarial_attack (bool): Whether to apply adversarial attack or not.
    - classifier (object): The classifier object to evaluate.

    Methods:
    - evaluate_prequential(window_size=1000): Evaluates the classifier using the prequential evaluation method.
    - plot_accuracy(): Plots the accuracy history of the classifier.

    Attributes:
    - classifier (object): The classifier object.
    - task_type (str): The type of task.
    - adversarial_attack (bool): Whether adversarial attack is applied or not.
    - dataset (Dataset): The dataset object.
    - accuracy_stream (list): List of accuracy values during evaluation.
    - overall_accuracy (float): The overall accuracy of the classifier.

    '''
    def __init__(self, task_type, dataset, adversarial_attack, classifier):
        self.classifier = classifier
        self.task_type = task_type
        self.adversarial_attack = adversarial_attack
        self.dataset = Dataset(
            dataset=dataset, adversarial_attack=adversarial_attack, save=False)
        if self.task_type == "classical":
            self.evaluate_prequential()
            self.plot_accuracy()



    def evaluate_prequential(self, window_size=1000):
        '''
        Evaluates the classifier using the prequential evaluation method.

        Parameters:
        - window_size (int): The size of the rolling window for calculating accuracy.

        '''
        classifier = self.classifier
        data = self.dataset.data
        target = self.dataset.label
        n_samples = data.shape[0]
        correct_cnt = 0
        self.accuracy_stream = []
        rolling_window = []
        window_correct_cnt = 0

        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"{i}/{n_samples}")
            X, y = data[i].reshape(1, -1), np.array([target[i]])
            y_pred = classifier.predict(X)
            classifier.partial_fit(X, y)
            correct = int(y == y_pred)
            rolling_window.append(correct)
            
            if len(rolling_window) > window_size:
                window_correct_cnt -= rolling_window.pop(0)
            
            window_correct_cnt += correct
            rolling_accuracy = window_correct_cnt / min(window_size, i + 1)
            self.accuracy_stream.append(rolling_accuracy)
            correct_cnt += correct

        self.overall_accuracy = correct_cnt / n_samples


    def plot_accuracy(self):
        '''
        Plots the accuracy history of the classifier.
        '''
        accuracy_history = self.accuracy_stream
        plt.figure(figsize=(10, 5), dpi=600)
        plt.plot(accuracy_history, '-bo', markersize=3)
        plt.xlabel('Time (in thousands of instances)')
        plt.ylabel('Accuracy')
        plt.show()
#%% SEA, AGRAWAL, spam and elec dataset performances on the HoeddinfTreeClassifier and AdaptiveRandomForest
stream_s2 = DataStream("classical", "SEA", False,AdaptiveRandomForest(n_estimators=5))
stream_s4 = DataStream("classical", "SEA", False,HoeffdingTreeClassifier())

stream_a2 = DataStream("classical", "AGRAWAL", False,AdaptiveRandomForest(n_estimators=5))
stream_a4 = DataStream("classical", "AGRAWAL", False,HoeffdingTreeClassifier())

stream_sp2 = DataStream("classical", "spam", False,AdaptiveRandomForest(n_estimators=5))
stream_sp4 = DataStream("classical", "spam", False,HoeffdingTreeClassifier())

stream_e2 = DataStream("classical", "elec", False,AdaptiveRandomForest(n_estimators=5))
stream_e4 = DataStream("classical", "elec", False,HoeffdingTreeClassifier())
#%% Plotting SEA dataset
xrange = np.array([i for i in range(1,len(stream_s2.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_s4.accuracy_stream)
plt.plot(xrange,stream_s2.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'AdaptiveRandomForest w/ 5 trees'])
#Plotting AGRAWAL dataset
xrange = np.array([i for i in range(1,len(stream_a2.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_a4.accuracy_stream)
plt.plot(xrange,stream_a2.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'AdaptiveRandomForest w/ 5 trees'])
#Plotting spam dataset
xrange = np.array([i for i in range(1,len(stream_sp2.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_sp4.accuracy_stream)
plt.plot(xrange,stream_sp2.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'AdaptiveRandomForest w/ 5 trees'])
#Plotting elec dataset
xrange = np.array([i for i in range(1,len(stream_e2.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_e4.accuracy_stream)
plt.plot(xrange,stream_e2.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'AdaptiveRandomForest w/ 5 trees'])

#%% Defining a custom ensemble model for the MajorityAdaptiveClassifier

#%%
class EnsembleBaseModel:
    '''
    A base class for ensemble models.

    Attributes:
        predictor: The underlying classifier used for prediction.
        is_trained: A flag indicating whether the model has been trained.
        feature_set: The selected subset of features used for training and prediction.

    Methods:
        __init__(self, feature_count): Initializes the EnsembleBaseModel object.
        partial_fit(self, X, y): Performs partial fit on the model.
        predict(self, X): Predicts the labels for the given input data.
        reset(self): Resets the model to its initial state.
    '''
    def __init__(self, feature_count):
        self.predictor = HoeffdingTreeClassifier() 
        self.is_trained = False  
        features = np.random.permutation(feature_count)
        self.feature_set = features[:feature_count * 9 // 10]

    def partial_fit(self, X, y):
        '''
        Fits the model incrementally on a batch of samples.

        Parameters:
            X (array-like): The input samples.
            y (array-like): The target values.

        Returns:
            None
        '''
        if not self.is_trained:
            self.predictor.partial_fit(X[:, self.feature_set], y, classes=[0, 1])
            self.is_trained = True
        else:
            self.predictor.partial_fit(X[:, self.feature_set], y, classes=[0, 1])

    def predict(self, X):
        return self.predictor.predict(X[:, self.feature_set])
    
    def reset(self):
        self.predictor = HoeffdingTreeClassifier()  
        self.is_trained = False  
        print("Resetted!")

class MajorityAdaptiveClassifier:
    '''
    A majority adaptive classifier that combines multiple base learners using an ensemble approach.
    
    Parameters:
    - feature_count (int): The number of features in the input data.
    - n_estimators (int, optional): The number of base learners in the ensemble. Default is 10.
    - delta (float, optional): The delta parameter for the ADWIN change detector. Default is 0.002.
    - detect_margin (int, optional): The margin for detecting change in the ADWIN change detector. Default is 3.
    '''
    def __init__(self, feature_count, n_estimators=10, delta=0.002, detect_margin=3):
        self.base_learners = [EnsembleBaseModel(feature_count) for _ in range(n_estimators)]
        self.weights = np.ones(n_estimators)
        self.n_estimators = n_estimators
        self.detect_margin = detect_margin
        self.detect_count = np.zeros(n_estimators)
        self.detectors = [ADWIN(delta=delta) for _ in range(n_estimators)]
        self.reset_requests = np.zeros(n_estimators)

    def partial_fit(self, X, y):
        '''
        Update the ensemble of base learners with new training data.
        
        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).
        - y (array-like): The target values of shape (n_samples,).
        '''
        for i, learner in enumerate(self.base_learners):
            initial_pred = learner.predict(X)
            learner.partial_fit(X, y)
            error = int(y != initial_pred[0])
            self.detectors[i].add_element(error)
            if self.detectors[i].detected_change():
                self.reset_requests[i] += 1
            else:
                self.reset_requests[i] += 0

        if np.sum(self.reset_requests > 0) >= self.n_estimators / 2:
            self.reset_all()

        for i, learner in enumerate(self.base_learners):
            if initial_pred == y:
                self.weights[i] *= 1.01
            else:
                self.weights[i] *= 0.99

    def predict(self, X):
        '''
        Predict the class labels for the input data.
        
        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).
        
        Returns:
        - y_pred (int): The predicted class label.
        '''
        predictions = np.array([learner.predict(X) for learner in self.base_learners])
        votes = np.bincount(predictions.reshape((1, -1))[0], weights=self.weights, minlength=2)
        return np.argmax(votes)

    def reset_all(self):
        '''
        Reset all base learners in the ensemble.
        '''
        for learner in self.base_learners:
            learner.reset()
        self.reset_requests.fill(0)
        print("Majority reset - all learners resetted!")

#%% Testing the custom ensemble classifier on the SEA, AGRAWAL, spam and elec datasets
stream_2s1 = DataStream("classical", "SEA", False,MajorityAdaptiveClassifier(3,n_estimators=5, delta=0.01,detect_margin=6))
stream_s4 = DataStream("classical", "SEA", False,HoeffdingTreeClassifier())

stream_2a1 = DataStream("classical", "AGRAWAL", False,MajorityAdaptiveClassifier(9,n_estimators=5, delta=0.01,detect_margin=7))
stream_a4 = DataStream("classical", "AGRAWAL", False,HoeffdingTreeClassifier())

stream_2sp1 = DataStream("classical", "spam", False,MajorityAdaptiveClassifier(499,n_estimators=5, delta=0.01,detect_margin=4))
stream_sp4 = DataStream("classical", "spam", False,HoeffdingTreeClassifier())

stream_2e1 = DataStream("classical", "elec", False,MajorityAdaptiveClassifier(6,n_estimators=5, delta=0.002,detect_margin=4))
stream_e4 = DataStream("classical", "elec", False,HoeffdingTreeClassifier())

#%% Plotting SEA dataset
xrange = np.array([i for i in range(1,len(stream_s4.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_s4.accuracy_stream)
plt.plot(xrange,stream_2s1.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'Custom Ensemble Classifier'])

# Plotting AGRAWAL dataset
xrange = np.array([i for i in range(1,len(stream_a4.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_a4.accuracy_stream)
plt.plot(xrange,stream_2a1.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'Custom Ensemble Classifier'])
#Plotting spam dataset
xrange = np.array([i for i in range(1,len(stream_sp4.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_sp4.accuracy_stream)
plt.plot(xrange,stream_2sp1.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'Custom Ensemble Classifier'])
#Plotting elec dataset
xrange = np.array([i for i in range(1,len(stream_e4.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_e4.accuracy_stream)
plt.plot(xrange,stream_2e1.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'Custom Ensemble Classifier'])

#%% Rewriting the DataStream class to evaluate adversarial attacks
class DataStream:
    '''
    Class representing a data stream for evaluating classifiers.
    '''
    def __init__(self, task_type, dataset, adversarial_attack, classifier):
        self.classifier = classifier
        self.task_type = task_type
        self.adversarial_attack = adversarial_attack
        self.dataset = Dataset(dataset=dataset, adversarial_attack=adversarial_attack, save=False)
        if self.task_type == "classical":
            self.evaluate_prequential()
        if self.adversarial_attack:
            self.evaluate_prequential_v2()
        self.plot_accuracy()

    
    def evaluate_prequential(self, window_size=1000):
        '''
        Method to evaluate the classifier using prequential evaluation on classical data.
        '''
        classifier = self.classifier
        data = self.dataset.data
        target = self.dataset.label
        n_samples = data.shape[0]
        correct_cnt = 0

        self.accuracy_stream = []
        rolling_window = []
        window_correct_cnt = 0
        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"{i}/{n_samples}")
            X, y = data[i].reshape(1, -1), np.array([target[i]])
            y_pred = classifier.predict(X)
            classifier.partial_fit(X, y)
            
            correct = int(y == y_pred)
            
            rolling_window.append(correct)
            if len(rolling_window) > window_size:
                window_correct_cnt -= rolling_window.pop(0)
            
            window_correct_cnt += correct
            rolling_accuracy = window_correct_cnt / min(window_size, i + 1)
            self.accuracy_stream.append(rolling_accuracy)
            
            correct_cnt += correct

        self.overall_accuracy = correct_cnt / n_samples

    
    def evaluate_prequential_v2(self, window_size=1000):
        '''
        Method to evaluate the classifier using prequential evaluation on adversarial data.
        '''
        classifier = self.classifier
        data = self.dataset.data
        target = self.dataset.label
        n_samples = data.shape[0]
        correct_cnt = 0

        self.accuracy_stream = []
        rolling_window = []

        window_correct_cnt = 0
        # Inject adversarial attacks during the training phase
        train_data, train_labels = inject_adversarial_attacks(data.copy(), target.copy(), attack_points=[40000, 60000], flip_percentage=[0.1, 0.2])
        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"{i}/{n_samples}")
            X, y = data[i].reshape(1, -1), np.array([target[i]])  # Correct label for testing
            X_train, y_train = data[i].reshape(1, -1), np.array([train_labels[i]])  # Possibly modified label for training
            y_pred = classifier.predict(X)
            
            classifier.partial_fit(X_train, y_train)
            
            correct = int(y == y_pred)
            
            rolling_window.append(correct)
            if len(rolling_window) > window_size:
                window_correct_cnt -= rolling_window.pop(0)
            
            window_correct_cnt += correct
            rolling_accuracy = window_correct_cnt / min(window_size, i + 1)
            self.accuracy_stream.append(rolling_accuracy)
            
            correct_cnt += correct

        self.overall_accuracy = correct_cnt / n_samples

    
    def plot_accuracy(self):
        '''
        Method to plot the accuracy stream.
        '''
        accuracy_history = self.accuracy_stream
        plt.figure(figsize=(10, 5), dpi=600)
        plt.plot(accuracy_history, '-bo', markersize=3)
        plt.xlabel('Time (in thousands of instances)')
        plt.ylabel('Accuracy')
        plt.show()
        
#%% Testing the custom ensemble classifier on the SEA dataset under adversarial attacks
stream_s5 = DataStream("adversarial", "SEA", True, MajorityAdaptiveClassifier(3,n_estimators=5, delta=0.01,detect_margin=6))
stream_s6 = DataStream("adversarial", "SEA", True, HoeffdingTreeClassifier())
# Plotting the results 
xrange = np.array([i for i in range(1,len(stream_s6.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_s6.accuracy_stream)
plt.plot(xrange,stream_s5.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'Custom Ensemble Classifier'])
#%% Testing the custom ensemble classifier on the AGRAWAL dataset under adversarial attacks
stream_a5 = DataStream("adversarial", "AGRAWAL", True, MajorityAdaptiveClassifier(6,n_estimators=5, delta=0.01,detect_margin=6))
stream_a6 = DataStream("adversarial", "AGRAWAL", True, HoeffdingTreeClassifier())
# Plotting the results 
xrange = np.array([i for i in range(1,len(stream_s6.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_s6.accuracy_stream)
plt.plot(xrange,stream_s5.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'Custom Ensemble Classifier'])

#%%
from sklearn.ensemble import IsolationForest

class EnsembleBaseModel:
    '''
    A base class for an ensemble model that combines a Hoeffding Tree classifier with an ADWIN change detector
    and an Isolation Forest for adversarial detection.

    Parameters:
    - delta (float): The delta parameter for the ADWIN change detector.
    - detect_margin (int): The number of detected changes required to trigger a reset.

    Attributes:
    - predictor: The Hoeffding Tree classifier used for prediction.
    - detector: The ADWIN change detector used for detecting changes.
    - isolation_forest: The Isolation Forest used for detecting adversarial examples.
    - delta (float): The delta parameter for the ADWIN change detector.
    - detect_count (int): The count of detected changes.
    - detect_margin (int): The number of detected changes required to trigger a reset.
    - is_trained (bool): Indicates whether the model has been trained or not.

    Methods:
    - partial_fit(X, y): Updates the model with new training instances.
    - reset(): Resets the model by reinitializing the predictor, detector, and detect_count.
    - predict(X): Predicts the class labels for the given instances.

    Example usage:
    ```
    model = EnsembleBaseModel(delta=0.001, detect_margin=5)
    model.partial_fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    '''
    def __init__(self, delta, detect_margin):
        self.predictor = HoeffdingTreeClassifier()  
        self.detector = ADWIN(delta=delta)
        self.isolation_forest = IsolationForest(n_estimators=10,contamination=0.01, random_state=42)
        self.delta = delta
        self.detect_count = 0
        self.detect_margin = detect_margin
        self.is_trained = False 

    def partial_fit(self, X, y):
        '''
        Partially fits the model with new training data.

        Parameters:
        X (array-like): The input samples.
        y (array-like): The target values.

        Returns:
        None
        '''
        if not self.is_trained:
            self.predictor.partial_fit(X, y, classes=[0,1])
            self.isolation_forest.fit(X)  # Train Isolation Forest on the initial benign data
            self.is_trained = True
        else:
            # Detect adversarial examples
            if np.any(self.isolation_forest.predict(X) == -1):  # -1 indicates anomaly
                print("Adversarial example detected!")
                return
            
            pred = self.predictor.predict(X)
            error = int(y != pred[0])
            self.predictor.partial_fit(X, y, classes=[0,1])
            self.detector.add_element(error)
            if self.detector.detected_change():
                self.detect_count += 1
            if self.detect_count >= self.detect_margin:
                self.reset()

    def reset(self):
        self.detector = ADWIN(delta=self.delta)
        self.predictor = HoeffdingTreeClassifier()  
        self.detect_count = 0
        print("Resetted!")

    def predict(self, X):
        return self.predictor.predict(X)

class MajorityAdaptiveClassifier:
    '''
    A majority adaptive classifier that combines multiple base learners to make predictions.

    Parameters:
    - n_estimators (int): The number of base learners in the ensemble. Default is 10.
    - delta (float): The threshold for detecting concept drift. Default is 0.002.
    - detect_margin (int): The margin for detecting concept drift. Default is 3.
    '''

    def __init__(self, n_estimators=10, delta=0.002, detect_margin=3):
        '''
        Initializes a MajorityAdaptiveClassifier object.

        Parameters:
        - n_estimators (int): The number of base learners in the ensemble. Default is 10.
        - delta (float): The threshold for detecting concept drift. Default is 0.002.
        - detect_margin (int): The margin for detecting concept drift. Default is 3.
        '''
        self.base_learners = [EnsembleBaseModel(delta, detect_margin) for _ in range(n_estimators)]
        self.weights = np.ones(n_estimators)
        self.n_estimators = n_estimators

    def partial_fit(self, X, y):
        '''
        Updates the ensemble of base learners with new training data.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).
        - y (array-like): The target values of shape (n_samples,).

        Returns:
        None
        '''
        for i, learner in enumerate(self.base_learners):
            initial_pred = learner.predict(X)
            learner.partial_fit(X, y)
            if initial_pred == y:
                self.weights[i] *= 1.05 
            else:
                self.weights[i] *= 0.95 

    def predict(self, X):
        '''
        Predicts the class labels for the input samples.

        Parameters:
        - X (array-like): The input features of shape (n_samples, n_features).

        Returns:
        - y_pred (int): The predicted class label.
        '''
        predictions = np.array([learner.predict(X) for learner in self.base_learners])
        votes = np.bincount(predictions.reshape((1, -1))[0], weights=self.weights, minlength=2)
        return np.argmax(votes)
    
#%% Testing the custom ensemble classifier on the SEA dataset under adversarial attacks
stream_s5 = DataStream("adversarial", "SEA", True, MajorityAdaptiveClassifier(n_estimators=5, delta=0.01,detect_margin=6))
stream_s6 = DataStream("adversarial", "SEA", True, HoeffdingTreeClassifier())
# Plotting the results 
xrange = np.array([i for i in range(1,len(stream_s6.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_s6.accuracy_stream)
plt.plot(xrange,stream_s5.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'Custom Ensemble Classifier'])
#%% Testing the custom ensemble classifier on the AGRAWAL dataset under adversarial attacks
stream_a5 = DataStream("adversarial", "AGRAWAL", True, MajorityAdaptiveClassifier(n_estimators=5, delta=0.01,detect_margin=6))
stream_a6 = DataStream("adversarial", "AGRAWAL", True, HoeffdingTreeClassifier())
#Plotting the results 
xrange = np.array([i for i in range(1,len(stream_s6.accuracy_stream)+1)])*1000
plt.figure(dpi = 100,figsize = (20,10))
plt.plot(xrange,stream_s6.accuracy_stream)
plt.plot(xrange,stream_s5.accuracy_stream)
plt.ylabel("Accuracy")
plt.xlabel("Instance")
plt.legend(['HoeffdingClassification Tree',
            'Custom Ensemble Classifier'])
