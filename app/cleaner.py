import pandas as pd
from pandas.api.types import is_object_dtype

class Cleaner:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Cleaner with a pandas DataFrame.
        """
        self.data = data

    def clean(self) -> pd.DataFrame:
        """
        Clean the DataFrame by:
        - Filling missing categorical/object values with the mode (most common value),
          or 'Missing' if the whole column is empty.
        - Filling missing numerical values with the mean,
          or 0 if the whole column is empty.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        for column in self.data.columns:
            if is_object_dtype(self.data[column]):
                # Handle categorical/object columns
                if self.data[column].isnull().all():
                    self.data[column] = self.data[column].fillna('Missing')
                else:
                    most_common = self.data[column].mode()[0]
                    self.data[column] = self.data[column].fillna(most_common)
            else:
                # Handle numerical columns
                if self.data[column].isnull().all():
                    self.data[column] = self.data[column].fillna(0)
                else:
                    mean_val = self.data[column].mean()
                    self.data[column] = self.data[column].fillna(mean_val)

        return self.data