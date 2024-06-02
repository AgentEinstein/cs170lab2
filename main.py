import random
import itertools
import math
import numpy as np
import time
from classifier import Classifier
from validator import Validator

def load_dataset(filename):
    data = np.loadtxt(filename)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return features, labels

def greedyforwardsearch(numfeatures):
    features = list(range(1, numfeatures + 1))
    bestaccuracy = {}
    allvisited = {}
    required_features = []

    for r in range(1, numfeatures + 1):
        accuracyoffeatures = {}
        maxsofar = -math.inf

        # Generate combinations of the remaining features
        remaining_features = [f for f in features if f not in required_features]
        for combination in itertools.combinations(remaining_features, r - len(required_features)):
            full_combination = tuple(sorted(combination + tuple(required_features)))
            
            if full_combination in allvisited:
                continue
            
            accuracy = random.uniform(0, 100)
            accuracyoffeatures[full_combination] = accuracy
            allvisited[full_combination] = accuracy
            
            if accuracy > maxsofar:
                maxsofar = accuracy
                bestaccuracy[full_combination] = maxsofar
            
            if r < 2:
                print(f'Using feature(s) {full_combination} accuracy is {accuracy:.2f}%')
        
        if r >= 2:
            updated = find_combinations_with_features(accuracyoffeatures, *required_features)
            for i, acc in updated.items():
                print(f'Using feature(s) {i} accuracy is {acc:.2f}%')
        
        print("\n")
        
        if r < 2:
            max_features = max(accuracyoffeatures, key=accuracyoffeatures.get)
            max_acc = max(accuracyoffeatures.values())
            print(f"Feature set {max_features} was best, accuracy is {max_acc:.2f}% \n")
            required_features = list(max_features)  # Update required features with the best set found
        
        elif r >= 2:
            max_features = max(updated, key=updated.get)
            max_acc = max(updated.values())
            print(f"Feature set {max_features} was best, accuracy is {max_acc:.2f}% \n")
            required_features = list(max_features)  # Update required features with the best set found
            if r == numfeatures:
                if max_acc < max(bestaccuracy.values()):
                    print("(Warning, Accuracy has decreased!)")
    
    print(f"Finished search!! The best feature subset is {max(bestaccuracy, key=bestaccuracy.get)}, which has an accuracy of {max(bestaccuracy.values()):.2f}%")

def backwardsearch(numfeatures):
    features = list(range(1, numfeatures + 1))  # Create a list of features, e.g., [1, 2, 3, 4] for 4 features
    bestaccuracy = {}
    allvisited = []
    subset_features = []

    for r in range(numfeatures, 0, -1):
        accuracyoffeatures = {}
        maxsofar = -math.inf
        required_features = [f for f in features if f not in subset_features]
        for combination in itertools.combinations(required_features, r):
            accuracy = random.uniform(0, 100)
            accuracyoffeatures[combination] = accuracy
            allvisited.append(combination)
            if accuracy > maxsofar:
                maxsofar = accuracy
                bestaccuracy[combination] = maxsofar
            if r == numfeatures:
                print(f'Using feature(s) {combination} accuracy is {accuracy:.2f}%')
        
        if r < numfeatures:
            updated = find_combinations_with_features_set(accuracyoffeatures, *required_features)
            for i, acc in updated.items():
                print(f'Using feature(s) {i} accuracy is {acc:.2f}%')

        print("\n")
        if r == numfeatures:
            max_features = max(accuracyoffeatures, key=accuracyoffeatures.get)
            max_acc = max(accuracyoffeatures.values())
            print(f"Feature set {max_features} was best, accuracy is {max_acc:.2f}% \n")
        elif r < numfeatures and updated:
            max_features = max(updated, key=updated.get)
            max_acc = max(updated.values())
            print(f"Feature set {max_features} was best, accuracy is {max_acc:.2f}% \n")
            if r == 1 and max_acc < max(bestaccuracy.values()):
                print("(Warning, Accuracy has decreased!)")
            
        if r >= 1:
            subset_features = [f for f in features if f not in max_features]


    print(f"Finished search!! The best feature subset is {max(bestaccuracy, key=bestaccuracy.get)}, which has an accuracy of {max(bestaccuracy.values()):.2f}%")

def special_algorithm(numfeatures):
    print("Choose the dataset to evaluate:")
    print(" 1 = Small dataset")
    print(" 2 = Large dataset")
    dataset_choice = int(input())

    if dataset_choice == 1:
        filename = 'data/small-test-dataset.txt'
    elif dataset_choice == 2:
        filename = 'data/large-test-dataset.txt'
    else:
        print("Invalid choice!")
        return

    data, labels = load_dataset(filename)

    # Feature subsets (example: use all features)
    feature_subset = list(range(data.shape[1]))

    print(f"The dataset has {numfeatures} features.")
    print("Enter the indices of the features you want to include (comma-separated), or press Enter to use all features:")
    user_input = input()

    if user_input.strip():
        feature_subset = [int(i) - 1 for i in user_input.split(',')]
    else:
        feature_subset = list(range(numfeatures))

    print(f"Using feature subset: {feature_subset}")

    # Initialize classifier and validator
    nn_classifier = Classifier()
    validator = Validator(nn_classifier)

    # Evaluate accuracy with tracing
    correct_predictions = 0
    num_instances = data.shape[0]
    trace = []

    for i in range(num_instances):
        start_time = time.time()
        train_data = np.delete(data, i, axis=0)
        train_labels = np.delete(labels, i)
        test_instance = data[i, :]
        test_label = labels[i]

        nn_classifier.train(train_data[:, feature_subset], train_labels)
        predicted_label = nn_classifier.test(test_instance[feature_subset])
        end_time = time.time()
        elapsed_time = end_time - start_time
        correct_predictions += (predicted_label == test_label)

        trace.append(f"Instance {i+1}: Predicted={predicted_label}, Actual={test_label}, Time={elapsed_time:.4f}s")

    accuracy = correct_predictions / num_instances
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Trace:")
    for line in trace:
        print(line)

def find_combinations_with_features(combinations, *features):
    # Filter combinations that contain all the specified features
    return {combination: accuracy for combination, accuracy in combinations.items() if all(feature in combination for feature in features)}

def find_combinations_with_features_set(combinations, *features):
    # Filter combinations that are subsets of the specified features
    return {combination: accuracy for combination, accuracy in combinations.items() if set(combination).issubset(features)}

print("Welcome to Group 45 Feature Selection Algorithm.\n")

print("Type the number of the algorithm you want to run.\n")
print(" 1 = Forward Selection")
print(" 2 = Backward Elimination")
print(" 3 = Special Algorithm\n")

choice = int(input())

print(f'Using no features and "random" evaluation, I get an accuracy of {random.uniform(0,100):.2f}% \n')
print("Beginning search.\n")
numfeatures = int(input("Please enter the total number of features: "))
if choice == 1:
    greedyforwardsearch(numfeatures)
elif choice == 2:
    backwardsearch(numfeatures)
elif choice == 3:
    special_algorithm(numfeatures)
