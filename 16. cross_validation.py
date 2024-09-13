from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC

iris = load_iris()

X, y = iris.data, iris.target

svm_classifier = SVC(kernel='linear', C=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(svm_classifier, X, y, cv=kfold)
print("Accuracy Scores for each fold: ", accuracy_scores)
print("Mean Accuracy: ", accuracy_scores.mean())