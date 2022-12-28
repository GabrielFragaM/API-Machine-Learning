from algorithms.algorithms import Algorithms
from fixers.define_values import DefineData
import pandas as pd
from fixers.treatment_values import TreatmentData


class MachineLearning:
    def __init__(self, base_x, base_y, test_size, algorithm_name, predict_csv, column_base_class, stand_scaler, one_hot_encoder, n_estimators_tree, criterion_tree):
        self.base_x = base_x
        self.base_y = base_y
        self.test_size = test_size
        self.algorithm_name = algorithm_name
        self.predict_csv = predict_csv
        self.column_base_class = column_base_class
        self.stand_scaler = stand_scaler
        self.one_hot_encoder = one_hot_encoder
        self.n_estimators_tree = n_estimators_tree
        self.criterion_tree = criterion_tree

    def start_machine_learning(self):

        def format_predict_values(predict_csv, column_base_class, stand_scaler, one_hot_encoder):
            csv = pd.read_csv(predict_csv)

            if column_base_class == '':
                cols = [c for c in csv.columns]
                column_base_class = cols[-1]

            treatment_data = TreatmentData(csv, column_base_class, stand_scaler, one_hot_encoder)
            predict_final = treatment_data.get_formatted_data()

            return predict_final

        x_training, x_test, y_training, y_test = DefineData.define_training_and_test_values(self.base_x, self.base_y, self.test_size)

        if self.predict_csv != '':
            predict_values = format_predict_values(self.predict_csv, self.column_base_class, self.stand_scaler, self.one_hot_encoder)
        else:
            predict_values = [x_test, y_test]

        algorithms = Algorithms(x_training, y_training, predict_values[1], predict_values[0], self.n_estimators_tree, self.criterion_tree)

        if self.algorithm_name == 'NaiveBayes':
            return algorithms.naive_bayes()
        if self.algorithm_name == 'TreeDecision':
            return algorithms.tree_decision()
        if self.algorithm_name == 'RandomForest':
            return algorithms.random_forest()



