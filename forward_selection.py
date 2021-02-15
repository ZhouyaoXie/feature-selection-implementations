import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

class ForwardSelection:
    
    def __init__(self, X, y, stopping_threshold = 4, verbose = True):
        """
        Initialize a ForwardSelection object.
        
        Parameters
        --------------
        X: Pandas DataFrame
            Training data
        
        y: array-like
            Response variables
        
        stopping_threshold: numeric, default is 4
            Threshold of F-ratio for when the algorithm stops. 
            
        verbose: bool
            Whether to print out a list of selected features at each round of training
            
        Returns
        --------------
        A ForwardSelection model object
        
        """
        self.stopping_threshold = stopping_threshold
        if type(X) != pd.DataFrame:
            print('Type Error: X is not a DataFrame.')
            exit()
        self.X = X
        self.y = np.array(y)
        self.verbose = verbose
        self.feature_set = []
        
    def forward(self):
        """
        Perform forward selection on X and y using F ratio.
        
        Returns
        --------------
        mdl: a statsmodels.regression.linear_model.RegressionResultsWrapper object
            The final fitted model using the selected features. 
            The list of selected features can be accessed using `mdl.params.index`
        
        """
        terminate = 0
        num_p = 0
        while terminate != 1:
            terminate = self.__add_feature()
            num_p+=1
            if self.verbose and terminate!=1:
                print('Round '+str(num_p)+': ', self.feature_set)
        return self.__OLS(self.X[self.feature_set], self.y)['model']
    
    def __OLS(self, X, y):
        """
        Perform OLS on input data and calculate its residual sum of squares.
        
        Returns
        ---------------
        re: dict
            A dictionary that contains keys "model" and "RSS"
        
        """
        model = sm.OLS(y,X).fit()
        RSS = np.sum((model.predict(X) - y) ** 2)
        return {"model":model, "RSS":RSS}
    
    def __add_feature(self):
        """
        Select the feature that will lead to the highest F-ratio increase.
        
        Returns
        -------------
        status: 0 or 1
            0 if a new feature is selected, 1 if no features with an F-ratio larger than 4 are found. 
            The algorithm stops when a 1 is returned.
        
        """
        remaining_predictors = [p for p in self.X.columns if p not in self.feature_set]
        if len(remaining_predictors) == self.X.shape[1]:
            RSS0 = np.sum(self.y ** 2)
        else:
            RSS0 = self.__OLS(self.X[self.feature_set], self.y)['RSS']
        F_max = 0
        F_max_feature = ''
        for p in remaining_predictors:
            m = self.__OLS(self.X[self.feature_set + [p]], self.y)
            F = (RSS0 - m['RSS']) / (m['RSS'] / m['model'].df_resid)
            if F > F_max:
                F_max = F
                F_max_feature = p
        if F_max < 4:
            return 1
        else:
            self.feature_set = self.feature_set + [F_max_feature]
            return 0
