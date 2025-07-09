class Trainer:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.prior = {}
        self.conditional_prob = {}

    def train(self):
        class_counts = self.data[self.target_column].value_counts()
        total_count = len(self.data)
        self.prior = (class_counts / total_count).to_dict()
        features = [col for col in self.data.columns if col != self.target_column]
        self.conditional_prob = {feature: {} for feature in features}
        for feature in features:
            feature_values = self.data[feature].unique()

            for value in feature_values:
                self.conditional_prob[feature][value] = {}

                for class_label in class_counts.index:
                    count = len(
                        self.data[(self.data[feature] == value) & (self.data[self.target_column] == class_label)])
                    class_count = class_counts[class_label]

                    num_unique_values = len(feature_values)
                    prob = (count + 1) / (class_count + num_unique_values)

                    self.conditional_prob[feature][value][class_label] = prob
                    return self.prior, self.conditional_prob