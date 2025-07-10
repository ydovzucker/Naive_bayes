from sklearn.model_selection import train_test_split
from typing import Dict

class ModelChecker:
    def __init__(self, prior: Dict, conditional_prob: Dict, target_column: str):
        self.prior = prior
        self.conditional_prob = conditional_prob
        self.target_column = target_column

    def predict(self, sample: Dict) -> str:
        class_scores = {}

        for class_label in self.prior:
            prob = self.prior[class_label]

            for feature, value in sample.items():
                if feature == self.target_column:
                    continue

                try:
                    cond_prob = self.conditional_prob[feature][value][class_label]
                except KeyError:
                    cond_prob = 1e-6  # Smoothing for unseen values

                prob *= cond_prob

            class_scores[class_label] = prob

        return max(class_scores, key=class_scores.get)

    def evaluate_accuracy(self, data):
        # Split into 70/30 train/test
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

        # Retrain on 70% data
        trainer = Trainer(train_data, self.target_column)
        prior, conditional_prob = trainer.train()

        # Create temporary checker for evaluation
        temp_checker = ModelChecker(prior, conditional_prob, self.target_column)

        correct = 0
        total = len(test_data)

        for _, row in test_data.iterrows():
            sample = row.drop(labels=[self.target_column]).to_dict()
            actual = row[self.target_column]
            prediction = temp_checker.predict(sample)
            if prediction == actual:
                correct += 1

        return correct / total