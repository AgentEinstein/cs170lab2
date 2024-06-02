class Classifier:
    def __init__(self):
        self.training_data = []
        self.labels = []

    def train(self, instances, labels):
        self.training_data = instances
        self.labels = labels

    def test(self, test_instance):
        distance = float('inf')
        label = None

        for i in range(len(self.training_data)):
            curr_instance = self.training_data[i]
            curr_distance = self.euclidean_distance(test_instance, curr_instance)

            if curr_distance < distance:
                distance = curr_distance
                label = self.labels[i]

        return label

    @staticmethod
    def euclidean_distance(point1, point2):
        distance = 0.0
        for p1, p2 in zip(point1, point2):
            distance += (p1 - p2) ** 2
        return distance ** 0.5
