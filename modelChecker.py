from sklearn.model_selection import train_test_split
class ModelChecker:
    def __init__(self, prior, conditional_prob, target_column):
        self.prior = prior
        self.conditional_prob = conditional_prob
        self.target_column = target_column

    def predict(self, sample):
        class_scores = {}

        for class_label in self.prior:
            # Start with the prior probability of the class
            prob = self.prior[class_label]

            # Multiply by the conditional probability of each feature value
            for feature, value in sample.items():
                if feature == self.target_column:
                    continue  # Skip the target column

                try:
                    cond_prob = self.conditional_prob[feature][value][class_label]
                except KeyError:
                    # If the value was not seen during training, skip or use small prob
                    cond_prob = 1e-6  # Tiny fallback

                prob *= cond_prob

            class_scores[class_label] = prob

        # Return the class with the highest score
        return max(class_scores, key=class_scores.get)

    def evaluate_accuracy(self, data):
        # Step 1: Split the data
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

        # Step 2: Train a new model on the 70% training data
        from trainer import Trainer  # (assuming this is the name of your Trainer class)
        trainer = Trainer(train_data, self.target_column)
        prior, conditional_prob = trainer.train()

        # Step 3: Create a temporary ModelChecker with the new training
        temp_checker = ModelChecker(prior, conditional_prob, self.target_column)

        # Step 4: Loop through test rows and make predictions
        correct = 0
        total = len(test_data)

        for _, row in test_data.iterrows():
            sample = row.drop(labels=[self.target_column]).to_dict()
            actual = row[self.target_column]
            prediction = temp_checker.predict(sample)

            if prediction == actual:
                correct += 1

        # Step 5: Return accuracy as a percentage
        accuracy = correct / total
        return accuracy