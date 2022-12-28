from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from fixers.helpers import get_index_column


class DefineData:
    def __init__(self, csv, column_base_class):
        self.csv = csv
        self.column_base_class = column_base_class

    def check_columns_to_encoder(self):
        def contains_strings(column_name):
            for v in self.csv[column_name]:
                if type(v) == str:
                    return column_name
                else:
                    return None

        columns_to_encoder = []

        for column in self.csv.axes[1]:
            if column != self.column_base_class:
                if contains_strings(column):
                    columns_to_encoder.append(column)

        return columns_to_encoder

    def define_forecasters(self):
        x_base = self.csv.iloc[:, 0:(len(self.csv.axes[1]) - 1)].values
        return x_base

    def define_class(self):
        base_y = self.csv.iloc[:, get_index_column(self.csv, self.column_base_class)].values
        return base_y

    @staticmethod
    def define_stand_scaler_values(base_x):
        base_x = StandardScaler().fit_transform(base_x)
        return base_x

    @staticmethod
    def define_encode(csv, base_x, columns_to_encoder):
        columns_indexes = []

        for c in columns_to_encoder:
            columns_indexes.append(get_index_column(csv, c))

        one_hot_encoder = ColumnTransformer(
            transformers=[('OneHot', OneHotEncoder(), columns_indexes)], remainder='passthrough')

        try:
            base_x = one_hot_encoder.fit_transform(base_x).toarray()
        except:
            base_x = one_hot_encoder.fit_transform(base_x)

        return base_x

    @staticmethod
    def define_string_values_to_number(csv, base_x, columns_to_encoder):
        if len(columns_to_encoder) != 0:
            for c in columns_to_encoder:
                base_x[:, get_index_column(csv, c)] = LabelEncoder(
                ).fit_transform(base_x[:, get_index_column(csv, c)])

            return base_x

    @staticmethod
    def define_training_and_test_values(base_x, base_y, test_size):
        return train_test_split(base_x, base_y, test_size=test_size, random_state=0)