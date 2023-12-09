import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from joblib import dump


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier with GridSearchCV to find the best parameters.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.

    Returns:
    RandomForestClassifier: The best estimator after performing grid search.
    """
    clf = RandomForestClassifier(random_state=153, verbose=False)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [20, 30, 50, 60, 70, None],
        'max_features': ['sqrt', 'log2'],
    }

    grid_search_clf = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted')
    grid_search_clf.fit(X_train, y_train)
    print(grid_search_clf.best_params_)
    print(grid_search_clf.best_score_)  
    return grid_search_clf.best_estimator_

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression Classifier with GridSearchCV to find the best parameters.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.

    Returns:
    LogisticRegression: The best estimator after performing grid search.
    """
    lr = LogisticRegression(verbose=False, max_iter=1000)

    c_values = np.concatenate((np.logspace(-4, 4, 20), np.arange(0.1, 1.0, 0.1)))

    param_grid = {
        'C': c_values,
        'fit_intercept': [False, True],
    }

    grid_search_lr = GridSearchCV(lr, param_grid, cv=5, scoring='f1_weighted')
    grid_search_lr.fit(X_train, y_train)
    print(grid_search_lr.best_params_)
    print(grid_search_lr.best_score_)
    return grid_search_lr.best_estimator_

def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting Classifier with GridSearchCV to find the best parameters.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.

    Returns:
    GradientBoostingClassifier: The best estimator after performing grid search.
    """
    gb = GradientBoostingClassifier()

    param_grid = {
        'n_estimators': [100, 200, 300],
    }

    grid_search_gb = GridSearchCV(gb, param_grid, cv=5, scoring='f1_weighted')
    grid_search_gb.fit(X_train, y_train)
    print(grid_search_gb.best_params_)
    print(grid_search_gb.best_score_)
    return grid_search_gb.best_estimator_

def train_gaussian_nb(X_train, y_train):
    """
    Train a Gaussian Naive Bayes Classifier with GridSearchCV to find the best parameters.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.

    Returns:
    GaussianNB: The best estimator after performing grid search.
    """
    gnb = GaussianNB()

    param_grid = {
        'var_smoothing': [1e-2, 1e-6, 1e-11, 1e-10, 1e-9],
    }

    grid_search_gnb = GridSearchCV(gnb, param_grid, cv=5, scoring='f1_weighted')
    grid_search_gnb.fit(X_train, y_train)
    print(grid_search_gnb.best_params_)
    print(grid_search_gnb.best_score_)
    return grid_search_gnb.best_estimator_

def train_bernoulli_nb(X_train, y_train):
    """
    Train a Bernoulli Naive Bayes Classifier with GridSearchCV to find the best parameters.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.

    Returns:
    BernoulliNB: The best estimator after performing grid search.
    """
    b_nb = BernoulliNB()

    param_grid = {
        'alpha': [1e-2, 1e-3, 0.5, 1],
        'binarize': [0.0, 0.5, 1.0],
        'fit_prior': [True, False],
    }

    grid_search_b_nb = GridSearchCV(b_nb, param_grid, n_jobs=-1, verbose=False, scoring='f1_weighted')
    grid_search_b_nb.fit(X_train, y_train)
    print(grid_search_b_nb.best_params_)
    print(grid_search_b_nb.best_score_)
    return grid_search_b_nb.best_estimator_

def save_best_model(model, file_name):
    """
    Save the best model to a file using joblib.

    Parameters:
    model (estimator): The trained model to save.
    file_name (str): The path to the file where the model should be saved.
    """
    dump(model, file_name)


