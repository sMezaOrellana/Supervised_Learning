from sklearn.svm import SVC
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import scale

def main():
    df = pd.read_csv ('data/train.csv')
    matrix = df.to_numpy()
    number_of_rows = matrix.shape[0]
    indices = np.random.choice(number_of_rows, size=20000, replace=False)

    matrix = matrix[indices]

    y = matrix[:,0]
    x = matrix[:,1:]
    X = x/255.0
    X_scaled = X
    #scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

    clf = SVC(C=100,gamma=0.01, kernel='rbf')
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

if __name__ == "__main__":
    main()