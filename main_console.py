import pandas as pd
from app.dataLoader import DataLoader
from app.cleaner import Cleaner
from app.config import target_column
from app.trainer import Trainer
from app.modelChecker import ModelChecker
from app.modelEvaluator import ModelEvaluator

def prompt_user_input(feature_names, data):
    """
    Prompt the user to input feature values with available options.
    """
    sample = {}
    print("\nEnter values for the following features:")
    for feature in feature_names:
        options = data[feature].dropna().unique()
        options_str = ", ".join(str(opt) for opt in options)
        value = input(f"{feature} (options: {options_str}): ")
        sample[feature] = value
    return sample

def main():
    print("=== Naive Bayes Classifier Console Interface ===")

    # Load and clean data
    loader = DataLoader()
    raw_data = loader.get_data()
    print(f"Loaded {len(raw_data)} rows of data.")

    cleaner = Cleaner(raw_data)
    data = cleaner.clean()

    # Get target column from user
    print("\nAvailable columns:")
    print(", ".join(data.columns))


    if target_column not in data.columns:
        print(f"Error: '{target_column}' not found in dataset columns.")
        return

    # Train model
    trainer = Trainer(data, target_column)
    prior, conditional_prob = trainer.train()
    model = ModelChecker(prior, conditional_prob, target_column)

    # Prepare evaluator
    evaluator = ModelEvaluator(model, target_column)

    feature_names = [col for col in data.columns if col != target_column]

    # Menu loop
    while True:
        print("\nOptions:")
        print("1. Predict new sample")
        print("2. Evaluate accuracy")
        print("3. Show confusion matrix")
        print("4. Exit")

        choice = input("Enter choice [1-4]: ")

        if choice == "1":
            sample = prompt_user_input(feature_names, data)
            prediction = model.predict(sample)
            print(f"\nPredicted class: {prediction}")

        elif choice == "2":
            accuracy = evaluator.evaluate_accuracy(data)
            print(f"\nAccuracy: {accuracy:.2%}")

        elif choice == "3":
            matrix = evaluator.compute_confusion_matrix(data)
            print("\nConfusion Matrix:")
            print(matrix)

        elif choice == "4":
            print("Exiting.")
            break

        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()