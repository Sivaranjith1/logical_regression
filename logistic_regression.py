import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, lr, num_of_weights):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.weights = np.random.rand(num_of_weights + 1, 1)
        self.lr = lr #Learning rate
        self.num_of_weights = num_of_weights
        
    def fit(self, X: pd.DataFrame, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        for index, row in X.iterrows():
            np_row = np.append(row.to_numpy(), [-1]) #Add bias of -1
            np_y = np.array([y[index]])

            for i in range(self.num_of_weights + 1):
                self.weights[i] += self.lr*np.matmul((np_y - self.predict_single(row)), [np_row[i]])
            
            #self.weights = self.weights + self.lr*np.matmul((np_y - self.predict(row).to_numpy()), np_row)
    def predict_single(self, X):
        np_row = np.append(X.to_numpy(), [-1]) #Add bias of -1
        return sigmoid(np.matmul(self.weights.transpose(), np_row))

    def predict(self, X: pd.DataFrame):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        output = []
        for index, row in X.iterrows():
            np_row = np.append(row.to_numpy(), [-1]) #Add bias of -1
            output.append(sigmoid(np.matmul(self.weights.transpose(), np_row))[0])
        return np.array(output)
        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        
