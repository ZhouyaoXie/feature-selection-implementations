import time
import pandas as pd
import numpy as np
import statsmodels.api as sm

class AllSubsets:
    
    def __init__(self, verbose = True, metric = "Mallow_Cp"):
        self.verbose = verbose
        self.metric = metric
        
    def __getBest(self, X, y, k):
        """
        Fit OLS models for all feature combinations.
        
        Returns
        ----------------------
        A dataframe of fitted models and their associated SSE values.
        
        Reference
        -----------------------
        http://www.science.smith.edu/~jcrouser/SDS293/labs/lab8-py.html
        
        """
    
        tic = time.time()
        
        results = []
        for combo in itertools.combinations(X.columns, k):
            results.append(self.__processSubset(X, y, combo))
        models = pd.DataFrame(results)
        
        toc = time.time()
        
        if self.verbose:
            print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
            
        return models

    def __processSubset(self, X, y, feature_set):
        """
        Fit OLS model and computes SSE.
        
        """
        model = sm.OLS(y, X[list(feature_set)])
        regr = model.fit()
        RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
        
        return {"model":regr, "RSS":RSS}
    
    def fit(self, X, y):
        """
        Fit a All-Subsets multiple linear regression model.
        
        Parameters
        ---------------------
        X: pandas DataFrame
            Column names are the feature names.
            
        y: numpy array
            Response variables.
            
        Returns
        ---------------------
        best_model: statsmodels.regression.linear_model.RegressionResultsWrapper
            One optimal model selected using Mallow's Cp
        
        """
        p = X.shape[1]
        m = X.shape[0]
        models_best = pd.DataFrame(columns=["RSS", "model", 'numb_features'])
        
        tic = time.time()
        
        for i in range(1, p+1):
            temp = self.__getBest(X, y, i)
            temp['numb_features'] = i
            models_best = models_best.append(temp)
            
        hat_sigma_squared = (1/(m - p -1)) * min(models_best['RSS'])
        models_best['C_p'] = (1/m) * (models_best['RSS'] + 2 * models_best['numb_features'] * hat_sigma_squared)
        best_model = models_best.iloc[np.argmin(models_best.C_p)]
            
        toc = time.time()
        print("Total elapsed time:", (toc-tic), "seconds.")
        
        return best_model
