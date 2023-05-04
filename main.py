import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
import matplotlib.pyplot as plt


# Read the data from the CSV file
df_train = pd.read_csv('train.csv', header=None)
df_test = pd.read_csv('test.csv', header=None)

# Split the training data into input features and labels
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]


X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define a list of classifiers to use
classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(),
    KMeans(n_clusters=2, n_init=10),
    MLPClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.2),
    LinearDiscriminantAnalysis(),
    GaussianProcessClassifier()
]

# Define a list of classifier names for plotting
classifier_names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'K-Nearest Neighbors',
    'Naive Bayes',
    'Support Vector Machine',
    'K-Means Clustering',
    'Multilayer Perceptron',
    'Linear Discriminant Analysis',
    'Gaussian Process'
]

# Train and evaluate each classifier
scores = []
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    scores.append([acc, f1, auc])

# Plot the results
scores = np.array(scores)
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, scores[:, 0], label='Accuracy')
plt.bar(classifier_names, scores[:, 1], label='F1 Score')
plt.bar(classifier_names, scores[:, 2], label='AUC Score')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Classifier')
plt.ylabel('Score')
plt.legend()
plt.show()

# Find the best classifier based on the highest average score
avg_scores = scores.mean(axis=1)
best_clf_idx = np.argmax(avg_scores)
best_clf_name = classifier_names[best_clf_idx]
print(f'The best classifier is {best_clf_name} with an average score of {avg_scores[best_clf_idx]:.4f}')

# Evaluate the best classifier on the testing set
best_clf = classifiers[best_clf_idx]
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'The best classifier achieved an accuracy of {acc:.4f} F')
