from dataLoader import DataLoader

loader = DataLoader()
df = loader.load_data("data/buy_computer_data.csv")
print(df.head())