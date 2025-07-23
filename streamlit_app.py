import streamlit as st
import pandas as pd
from app.dataLoader import DataLoader
from app.cleaner import Cleaner
from app.config import target_column
from app.trainer import Trainer
from app.modelChecker import ModelChecker
from app.modelEvaluator import ModelEvaluator

def main():
    st.title("Naive Bayes Classifier - Streamlit App")

    # Load and clean data
    loader = DataLoader()
    raw_data = loader.get_data()
    cleaner = Cleaner(raw_data)
    data = cleaner.clean()

    if target_column not in data.columns:
        st.error(f"'{target_column}' not found in dataset columns.")
        return

    # Train model
    trainer = Trainer(data, target_column)
    prior, conditional_prob = trainer.train()
    model = ModelChecker(prior, conditional_prob, target_column)

    # Prepare evaluator
    evaluator = ModelEvaluator(model, target_column)

    feature_names = [col for col in data.columns if col != target_column]

    # Sidebar navigation
    option = st.sidebar.selectbox("Select an option:", [
        "Predict new sample", "Evaluate accuracy", "Show confusion matrix"])

    if option == "Predict new sample":
        st.header("Predict a New Sample")
        user_input = {}
        for feature in feature_names:
            options = data[feature].dropna().unique()
            value = st.selectbox(f"{feature}", options)
            user_input[feature] = value

        if st.button("Predict"):
            prediction = model.predict(user_input)
            st.success(f"Predicted class: {prediction}")

    elif option == "Evaluate accuracy":
        st.header("Evaluate Accuracy")
        accuracy = evaluator.evaluate_accuracy(data)
        st.write(f"Accuracy: {accuracy:.2%}")

    elif option == "Show confusion matrix":
        st.header("Confusion Matrix")
        matrix = evaluator.compute_confusion_matrix(data)
        st.dataframe(matrix)

if __name__ == "__main__":
    main()
