from fixers.define_values import DefineData
from fixers.helpers import get_columns


class TreatmentData:
    def __init__(self, csv, column_base_class, stand_scaler, one_hot_encoder):
        self.csv = csv
        self.column_base_class = column_base_class
        self.stand_scaler = stand_scaler
        self.one_hot_encoder = one_hot_encoder

    def check_values_empty(self):
        columns = get_columns(self.csv)
        columns_null = self.csv.isnull().sum()
        index = 0
        for c in columns_null:
            if c != 0:
                try:
                    self.csv.fillna(self.csv[columns[index]].mean(), inplace=True)
                except:
                    self.csv = self.csv[self.csv[columns[index]].notna()]
                    self.csv = self.csv.reset_index(drop=True)
            index += 1
        return self.csv

    def get_formatted_data(self):

        if self.column_base_class == '':
            cols = [c for c in self.csv.columns]
            self.column_base_class = cols[-1]

        self.csv = self.check_values_empty()

        define_data = DefineData(self.csv, self.column_base_class)

        base_x = define_data.define_forecasters()
        base_y = define_data.define_class()
        columns_to_encoder = define_data.check_columns_to_encoder()

        if len(columns_to_encoder) != 0:
            base_x = define_data.define_string_values_to_number(self.csv, base_x, columns_to_encoder)
            if self.one_hot_encoder:
                base_x = define_data.define_encode(self.csv, base_x, columns_to_encoder)

        if self.stand_scaler:
            base_x = define_data.define_stand_scaler_values(base_x)

        return [base_x, base_y]

