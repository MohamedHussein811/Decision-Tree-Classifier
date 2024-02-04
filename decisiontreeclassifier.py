import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = load_digits()
dataset=pd.DataFrame(data=data['data'],columns=data['feature_names'])
dataset

X = dataset.copy()
Y = data['target']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.70)

classifier = DecisionTreeClassifier()
classifier =classifier.fit(X_train,Y_train)
classifier.get_params()

predictions = classifier.predict(X_test)
predictions

accuracy=accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy}')

confusion_matrix(Y_test,predictions,labels=[0,1])

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(data.images[89])
plt.show()

num_samples = X_test.shape[0]
num_cols = 5
num_rows = (num_samples + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 2 * num_rows))

axes = axes.flatten()

for i in range(num_samples):
    axes[i].bar(range(10), classifier.predict_proba(X_test.iloc[i].values.reshape(1, -1))[0])
    axes[i].set_xlabel('Digit Class')
    axes[i].set_ylabel('Probability')
    axes[i].set_title(f'Sample {i} - True Class: {Y_test[i]}')

plt.tight_layout()

plt.show()

plt.bar(range(10), classifier.predict_proba(X_test.iloc[89].values.reshape(1, -1))[0])
plt.xlabel('Digit Class')
plt.ylabel('Probability')
plt.title('Predicted Class Probabilities for Sample 89')
plt.xticks(range(10), [str(i) for i in range(10)])
plt.show()

feature_importance = pd.DataFrame(classifier.feature_importances_, index=X.columns, columns=['Importance'])
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
feature_importance

from matplotlib import pyplot as plt
feature_importance['Importance'].plot(kind='hist', bins=20, title='Importance')
plt.gca().spines[['top', 'right',]].set_visible(False)

from sklearn import tree
from matplotlib import pyplot as plt


fig, ax = plt.subplots(figsize=(25, 20))
tree.plot_tree(classifier, filled=True, feature_names=X.columns, class_names=True, ax=ax)

plt.show()