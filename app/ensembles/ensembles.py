import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


def MSE(y_true, y_pred):
    return np.mean(((y_true -  y_pred))**2)



class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None, estimator=DecisionTreeRegressor,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.feature_subsample_size = feature_subsample_size
        self.max_depth = max_depth
        self.trees_parameters = trees_parameters

        self.trees = []
        self.feature_samples = []
        self.estimator = estimator


    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = max(X.shape[1] // 3, 1)

        train_compos_pred = 0.0 #np.zeros(X.shape[0], dtype='float32')
        test_compos_pred = 0.0
        train_mses = []
        test_mses = []
        
        for i in range(self.n_estimators):
            train_sample = np.random.choice(X.shape[0], size=X.shape[0])
            self.feature_samples.append(np.random.choice(X.shape[1], size=self.feature_subsample_size, replace=False))
            bootstraped_sample = X[:, self.feature_samples[i]][train_sample]
            
            tree = self.estimator(max_depth=self.max_depth, **self.trees_parameters)
            tree.fit(bootstraped_sample, y[train_sample])
            self.trees.append(tree)
            train_compos_pred += (self.trees[i].predict(X[:, self.feature_samples[i]]))
            train_mses.append(MSE(y, train_compos_pred / (i + 1)))

            if X_val is not None:
                test_compos_pred += (self.trees[i].predict(X_val[:, self.feature_samples[i]]))
                test_mses.append(MSE(y_val, test_compos_pred / (i + 1)))
        
        return self, \
                {
                    'train_mses': train_mses,
                    'test_mses': test_mses
                }

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------:
        y : numpy ndarray
            Array of size n_objects
        """
        y_pred = 0.0
        for i in range(self.n_estimators):
            y_pred += self.trees[i].predict(X[:, self.feature_samples[i]])
        return y_pred / self.n_estimators

class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        estimator=DecisionTreeRegressor, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

        self.feature_samples = []
        self.trees = []
        self.weights = []
        self.estimator = estimator

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """


        if self.feature_subsample_size is None:
            self.feature_subsample_size = max(X.shape[1] // 3, 1)
        
       
        train_mses = []
        test_mses = []
        f = 0.0
        y = y.ravel()
        if y_val is not None:
            y_val = y_val.ravel()
    

        for i in range(self.n_estimators):
       #     print('zero')
            self.feature_samples.append(np.random.choice(X.shape[1], self.feature_subsample_size, replace=False))
            train_sample = np.random.choice(X.shape[0], X.shape[0])

            tree = self.estimator(max_depth=self.max_depth, **self.trees_parameters)
        #    print('1.5')
         #   print(f.shape if i else 0, y.shape)
            self.trees.append(tree.fit(X[:, self.feature_samples[i]][train_sample], (y - f)[train_sample])) # d (mse)/ dy  = 2 * (y - f)
          #  print('first')
            y_pred = self.trees[i].predict(X[:, self.feature_samples[i]])
            if i == 0:
                self.weights.append(1.0)
            else:
                self.weights.append(minimize_scalar(lambda w: np.mean((y - (f + w * y_pred))**2)).x * self.lr)
           # print('second')
            f += self.weights[i] * y_pred
            
            train_mses.append(MSE(y, f))
            #print('third')
            if X_val is not None:
                pred = 0.0
                for j in range(i + 1):
                    pred += self.weights[j] * self.trees[j].predict(X_val[:, self.feature_samples[j]])
                test_mses.append(MSE(y_val, pred))
 
        return self, \
            {
                'train_mses': train_mses,
                'test_mses': test_mses
            }
            
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        y_pred = 0.0
       
        for i in range(self.n_estimators):
            y_pred += self.weights[i] * self.trees[i].predict(X[:, self.feature_samples[i]])
        return y_pred
