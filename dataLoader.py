import pandas as pd

class DataLoader:
    def __init__(self):
        self.data = None

    def load_data(self, filepath="data/data.csv"):
        try:
            self.data = pd.read_csv(filepath)
            return self.data
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None