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