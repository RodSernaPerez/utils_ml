import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


class ClassificationComparator:
    '''
    Performs simple test to check and compare several classification
    methods
    '''

    def __init__(self, number_estimators_random_forests=10):
        self.operators = {'Random forest': RandomForestRegressor(number_estimators_random_forests),
                          'Logistic regression': LogisticRegression()}

    def run(self, x, y, test_percentage=0.2):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage)

        for name, operator in self.operators.items():
            operator.fit(x_train, y_train)
            predictions_of_test = np.round(operator.predict(x_test))

            print("MODEL {}".format(name))
            print("=========================")
            print("Accuracy: {}".format(
                accuracy_score(predictions_of_test, y_test)))
            print("F1 score: {}".format(
                f1_score(predictions_of_test, y_test)))
