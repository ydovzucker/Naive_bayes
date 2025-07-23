from typing import Dict
from app.config import target_column

class ModelChecker:
    """
    Uses learned prior and conditional probabilities to predict the class of a given sample
    using the Naive Bayes formula.
    """

    def __init__(self, prior: Dict[str, float], conditional_prob: Dict[str, Dict[str, Dict[str, float]]], target_column: str):
        """
        Initializes the model checker with prior and conditional probabilities.

        :param prior: A dictionary mapping class labels to their prior probabilities.
        :param conditional_prob: A nested dictionary of the form:
                                 {feature: {value: {class_label: probability}}}
        :param target_column: The name of the target column to ignore during prediction.
        """
        self.prior = prior
        self.conditional_prob = conditional_prob
        self.target_column = target_column

    def predict(self, sample: Dict[str, str]) -> str:
        """
        Predicts the class label for a given input sample using the Naive Bayes rule.

        :param sample: A dictionary representing a sample to classify.
                       Example: {"age": "30-40", "income": "high"}
        :return: The predicted class label with the highest posterior probability.
        """
        class_scores: Dict[str, float] = {}

        for class_label in self.prior:
            prob = self.prior[class_label]

            for feature, value in sample.items():
                if feature == self.target_column:
                    continue

                try:
                    cond_prob = self.conditional_prob[feature][value][class_label]
                except KeyError:
                    cond_prob = 1e-6  # Smoothing for unseen feature-values

                prob *= cond_prob

            class_scores[class_label] = prob

        return max(class_scores, key=class_scores.get)