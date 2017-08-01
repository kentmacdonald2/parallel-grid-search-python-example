from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# load dataset
digits = load_digits()

# declare decision tree
clf = DecisionTreeClassifier()

# get accuracy scores for each of the 10 folds
# scores = cross_val_score(clf, X=digits.data, y=digits.target, scoring='accuracy', cv=3)

# # store average accuracy score
# cv_result = sum(scores) / len(scores)

# #  print accuracy
# print("Result: " + (cv_result * 100).__str__() + "% accuracy")


# hyper-parameter grid
param_grid = {
    'criterion': ["gini", "entropy"],
    'max_depth': [None, 2, 4, 6, 8, 10, 12],
    'max_features': ['sqrt', 'log2', None, 2, 4, 6, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4]
}

# pass parameters to GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

# set X and y variables
grid_search.fit(X=digits.data, y=digits.target)

# get mean accuracy from grid search
means = grid_search.cv_results_['mean_test_score']

# get standard deviation from grid_search
stds = grid_search.cv_results_['std_test_score']

# print results
print("Best Parameters: " + grid_search.best_params_.__str__())
print("Best Score: " + grid_search.best_score_.__str__())
