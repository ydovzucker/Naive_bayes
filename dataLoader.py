import pandas as pd

class DataLoader:
    def __init__(self):
        self.data = None

    def load_data(self, source, source_type='csv', **kwargs):
        if source_type == 'csv':
            self.data = pd.read_csv(source, **kwargs)
        elif source_type == 'excel':
            self.data = pd.read_excel(source, **kwargs)
        elif source_type == 'sql':
            self.data = pd.read_sql(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")
        return self.data