import pandas as pd
from dataLoader import DataLoader
from cleaner import Cleaner
from trainer import Trainer
from modelChecker import ModelChecker
def get_user_input(features):
    sample = {}
    for feature in features:
        value = input(f"Enter value for '{feature}': ")
        sample[feature] = value
    return sample
def main():
    source_type = "csv"
    source = "data/data.csv"

    loader = DataLoader()
    data = loader.load_data(source, source_type=source_type)

    target_column = input("Enter the name of the target (label) column: ")

    cleaner = Cleaner(data)
    cleaned_data = cleaner.clean()

    trainer = Trainer(cleaned_data, target_column)
    prior, conditional_prob = trainer.train()

    checker = ModelChecker(prior, conditional_prob, target_column)

    features = [col for col in cleaned_data.columns if col != target_column]
    sample = get_user_input(features)

    prediction = checker.predict(sample)
    print(f"\nPredicted class: {prediction}")

    accuracy = checker.evaluate_accuracy(cleaned_data)
    print(f"Model accuracy (on random 30% test split): {accuracy:.2%}")