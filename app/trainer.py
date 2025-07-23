from typing import Dict, Tuple
import pandas as pd
from app.config import target_column

class Trainer:
    """
    Trainer class to calculate prior and conditional probabilities
    for a Naive Bayes classifier given cleaned data.
    """

    def __init__(self, data: pd.DataFrame, target_column: str):
        """
        Initialize Trainer with data and target column.

        Args:
            data (pd.DataFrame): Cleaned input data.
            target_column (str): Name of the target (label) column.
        """
        self.data = data
        self.target_column = target_column
        self.prior: Dict[str, float] = {}
        self.conditional_prob: Dict[str, Dict] = {}

        self._validate_data()

    def _validate_data(self):
        """
        Validate that the target column exists in the data.
        Raises:
            ValueError: If target_column not in data columns.
        """
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")

    def train(self) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """
        Calculate prior and conditional probabilities.

        Returns:
            prior (Dict[str, float]): Prior probabilities P(class).
            conditional_prob (Dict[str, Dict]): Conditional probabilities P(feature=value | class).
        """
        # Calculate prior probabilities P(class)
        class_counts = self.data[self.target_column].value_counts()
        total_count = len(self.data)
        self.prior = (class_counts / total_count).to_dict()

        # Identify features (exclude the target)
        features = [col for col in self.data.columns if col != self.target_column]
        self.conditional_prob = {feature: {} for feature in features}

        # Calculate conditional probabilities P(feature=value | class)
        for feature in features:
            feature_values = self.data[feature].unique()

            for value in feature_values:
                self.conditional_prob[feature][value] = {}

                for class_label in class_counts.index:
                    count = len(
                        self.data[(self.data[feature] == value) &
                                  (self.data[self.target_column] == class_label)]
                    )
                    class_count = class_counts[class_label]

                    # Laplace smoothing
                    num_unique_values = len(feature_values)
                    prob = (count + 1) / (class_count + num_unique_values)

                    self.conditional_prob[feature][value][class_label] = prob

        return self.prior, self.conditional_prob