import pandas as pd

class Cleaner:
    def __init__(self, data):
        self.data = data

    def clean(self):
        for column in self.data.columns:
            if pd.api.types.is_object_dtype(self.data[column]) or pd.api.types.is_categorical_dtype(self.data[column]):
                if self.data[column].isnull().all():
                    self.data[column] = self.data[column].fillna('Missing')
                else:
                    most_common = self.data[column].mode()[0]
                    self.data[column] = self.data[column].fillna(most_common)
            else:
                if self.data[column].isnull().all():
                    self.data[column] = self.data[column].fillna(0)
                else:
                    mean_val = self.data[column].mean()
                    self.data[column] = self.data[column].fillna(mean_val)
        return self.data