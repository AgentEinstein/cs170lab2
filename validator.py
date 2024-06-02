import numpy as np

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def evaluate(self, data, labels, feature_subset):
        correct_predictions = 0
        num_instances = data.shape[0]
        for i in range(num_instances):
            train_data = np.delete(data, i, axis=0)
            train_labels = np.delete(labels, i)
            test_instance = data[i, :]
            test_label = labels[i]

            self.classifier.train(train_data[:, feature_subset], train_labels)
            predicted_label = self.classifier.test(test_instance[feature_subset])
            if predicted_label == test_label:
                correct_predictions += 1

        accuracy = correct_predictions / num_instances
        return accuracy