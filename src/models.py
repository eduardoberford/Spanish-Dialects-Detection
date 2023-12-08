#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.exceptions import ConvergenceWarning

import numpy as np
import warnings
import sys
import os

class LogisticRegressionITDI:
    """
    LRITDI
    ------
    Model developed by ETHZ group in the context of ITDI Shared Task @ Vardial 2022.
    Link to the shared task : https://sites.google.com/view/vardial-2022/shared-tasks#h.x0q0staz3tdr
    The model is composed by three simple components:
        - TF-IDF vectorizer
        - Scaler
        - Logistic Regression classifier

    Parameters
    ----------
    random_state : int (default = 0)
                   seed to control the non deterministic components of the model.
    
    verbose : int (default = 0)
              set the verbosity level of the model, 0 for no output and 1 for verbose output.

    n_jobs : int (default = -1)
             Number of CPU cores used when parallelizing over classes. -1 means using all processors. 
    """

    def __init__(self, random_state: int = 0, verbose: int = 0, n_jobs: int = -1) -> None:

        self.model_name = "LRITDI"
        self.vectorizer = TfidfVectorizer()
        self.scaler =  StandardScaler(with_mean=False)
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = LogisticRegression(penalty='l2', 
                                        random_state=self.random_state, 
                                        n_jobs=self.n_jobs, 
                                        solver="sag", 
                                        class_weight={0:1, 1:1, 2:1, 3:1, 4:1},
                                        #class_weight={0:5.01329465, 1:1.59955319, 2:0.60311469, 3:2.60301423, 4:1.48266328, 5:1.8468792 , 6:0.41534923, 7:2.80706276, 8:0.5999105 , 9:0.65356942, 10:1.04561053},
                                        verbose=0
                                        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Tuple:
        """
        Fit the model with training data. If hold-out data are provided, evaluate the model.

        Parameters
        ----------
        X_train : np.ndarray 
                  input sentences array, shape (n_sentences, 1).

        y_train : np.ndarray 
                  labels for the training sentences, shape (n_sentences, 1).

        X_val : np.ndarray 
                validation sentences array, shape (n_sentences, 1). None to avoid validation.
        
        y_val : np.ndarray 
                  labels for the validation sentences, shape (n_sentences, 1). None to avoid validation.
        """
        
        # vectorize and scale training data
        if self.verbose: print("Vectorizing and normalizing training data...")
        X_train = self.vectorizer.fit_transform(X_train)
        X_train = self.scaler.fit_transform(X_train)

        # fit the model
        if self.verbose: print("Fitting Logistic Regression...")
        with warnings.catch_warnings(): 
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                self.model.fit(X_train, y_train)

        # predict for training
        y_pred = self.model.predict(X_train)

        # evaluate training score
        train_score = f1_score(y_train, y_pred, average="micro")
        val_score = np.nan
        cm = None

        # hold-out validation
        if X_val is not None:
            
            if self.verbose: print("Evaluating the model...")
            # vectorize and scale validation set
            X_val = self.vectorizer.transform(X_val)
            X_val = self.scaler.transform(X_val)

            # predict validation labels
            y_pred = self.model.predict(X_val)

            # score the predictions
            val_score = f1_score(y_val, y_pred, average="micro")
            cm = confusion_matrix(y_true=y_val, y_pred=y_pred, normalize="true", labels=range(0,5))


        if self.verbose: print(f"Training completed, train_f1 : {train_score}, val_f1 : {val_score}")

        return train_score, val_score, cm

    
    def predict(self, X):
        """
        Predict the dialect label for a given sample.

        Parameters
        ----------
        X : str, np.ndarray(n_sentences,)
            string or array of strings for which a dialect label must be predicted

        Returns
        -------
        y : int [0, 10], np.ndarray(nsentences,)
            predict the corresponding label/s for the given samples
        """
        if isinstance(X, str): X = [X]
        # vectorize and scale test data
        X = self.vectorizer.transform(X)
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        if isinstance(X, str): X = [X]
        # vectorize and scale test data
        X = self.vectorizer.transform(X)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def rkf_cv(self, X, y, folds = 5, n_repeats = 3):
        """
        Perform repeated k-fold cross-validation on the training data.

        Parameters
        ----------
        X : np.ndarray (n_sentences,)
            training data
        
        y : np.ndarray (n_sentences,)
            training labels
        
        folds : int (default=5)
                number of folds for k-fold CV
        
        n_repeats : int (default=3)
                    number of times for which the cross-validation must be repeated
        
        Returns
        -------
        mean : float
               mean validation f1-micro score
        
        std : float
              standard deviation across the validation f1-scores
        """
        X = self.vectorizer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"
            rkf = RepeatedKFold(n_splits=folds, n_repeats=n_repeats, random_state=self.random_state)
            scores = cross_val_score(self.model, X, y, scoring='f1_micro', cv=rkf, n_jobs=-1)
            return np.mean(scores), np.std(scores)
    
    
def main():

    from utils import load_data
    X, y = load_data(train_path="train.csv")
    print("Successfuly loaded X and y with shapes: " + str(X.shape) + str(y.shape))
    print("Cross-validating Logistic Regression...")
        
    model = LogisticRegressionITDI(verbose=1)
    mean, std = model.rkf_cv(X, y)
    
    print(f'Repeated cross-validation scores: {mean} ({std})')

if __name__ == "__main__":
    main()