import pandas as pd
from app.modelChecker import ModelChecker
from app.trainer import Trainer

class ModelEvaluator:
    """
    Evaluates a Naive Bayes model using accuracy and confusion matrix.

    Attributes:
        model_checker (ModelChecker): A pre-trained model used for evaluation.
        target_column (str): The name of the target column in the dataset.
    """

    def __init__(self, model_checker: ModelChecker, target_column: str):
        """
        Initializes the evaluator with a pre-trained model.

        :param model_checker: An instance of ModelChecker used for prediction.
        :param target_column: Name of the target column to evaluate.
        """
        self.model_checker = model_checker
        self.target_column = target_column

    def evaluate_accuracy(self, data: pd.DataFrame) -> float:
        """
        Trains a model on 70% of the data and tests it on 30%, returning the accuracy.

        :param data: Full dataset including features and target column.
        :return: Accuracy score as a float (0.0 to 1.0).
        """
        from sklearn.model_selection import train_test_split

        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

        trainer = Trainer(train_data, self.target_column)
        prior, conditional_prob = trainer.train()

        temp_model = ModelChecker(prior, conditional_prob, self.target_column)

        correct = 0
        for _, row in test_data.iterrows():
            sample = row.drop(labels=[self.target_column]).to_dict()
            actual = row[self.target_column]
            prediction = temp_model.predict(sample)
            if prediction == actual:
                correct += 1

        return correct / len(test_data)

    def compute_confusion_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the confusion matrix using the provided model.

        :param data: DataFrame containing samples and target column.
        :return: A pandas DataFrame representing the confusion matrix.
        """
        actuals = []
        predictions = []

        for _, row in data.iterrows():
            sample = row.drop(labels=[self.target_column]).to_dict()
            actual = row[self.target_column]
            prediction = self.model_checker.predict(sample)
            actuals.append(actual)
            predictions.append(prediction)

        return pd.crosstab(
            pd.Series(actuals, name="Actual"),
            pd.Series(predictions, name="Predicted"),
            rownames=["Actual"],
            colnames=["Predicted"],
            dropna=False
        )