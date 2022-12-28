import pandas as pd

from fixers.treatment_values import TreatmentData
from machine.machine_learning import MachineLearning


def main(csv: str, column_base_class: str, stand_scaler: bool,
         one_hot_encoder: bool, test_size: float, algorithm_name: str,
         predict_csv: str, n_estimators_tree: int, criterion_tree: str):

    csv = pd.read_csv(csv)

    if n_estimators_tree == 0:
        n_estimators_tree = 30

    if criterion_tree == '':
        criterion_tree = 'entropy'


    treatment_data = TreatmentData(csv, column_base_class, stand_scaler, one_hot_encoder)

    base = treatment_data.get_formatted_data()
    machine_learning = MachineLearning(base[0], base[1], test_size, algorithm_name, predict_csv, column_base_class,
                                       stand_scaler, one_hot_encoder, n_estimators_tree, criterion_tree)

    return machine_learning.start_machine_learning()

###Algorithms###
#NaiveBayes
#TreeDecision = criterion_tree, n_estimators_tree
#RandomForest = criterion_tree, n_estimators_tree

print(main(
    csv='census.csv', column_base_class='income', stand_scaler=False,
    one_hot_encoder=False, test_size=0.15,
    algorithm_name='RandomForest',
    predict_csv='census_test.csv',
    n_estimators_tree=30, criterion_tree='entropy'))