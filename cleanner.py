class Cleaner:
    def __init__(self, data):
        self.data = data

    def clean(self):
        for column in self.data.columns:
            if self.data[column].dtype == 'object':  # Categorical
                # Fill missing with mode or add 'Missing' if mode not found
                if self.data[column].isnull().all():
                    self.data[column] = self.data[column].fillna('Missing')
                else:
                    most_common = self.data[column].mode()[0]
                    self.data[column] = self.data[column].fillna(most_common)
            else:  # Numeric
                # Fill missing with mean (if numeric column is not empty)
                if self.data[column].isnull().all():
                    self.data[column] = self.data[column].fillna(0)  # Or some constant
                else:
                    mean_val = self.data[column].mean()
                    self.data[column] = self.data[column].fillna(mean_val)
        return self.data