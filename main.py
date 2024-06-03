import itertools
import math
import random
import numpy as np
from classifier import Classifier
from validator import Validator

def load_dataset(filename):
    data = np.loadtxt(filename)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    epsilon = 1e-8 
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + epsilon)
    return features, labels

def evaluate_with_leave_one_out(data, labels, feature_subset, classifier):
    correct_predictions = 0
    num_instances = data.shape[0]
    for i in range(num_instances):
        train_data = np.delete(data, i, axis=0)
        train_labels = np.delete(labels, i)
        test_instance = data[i]
        test_label = labels[i]
        classifier.train(train_data[:, feature_subset], train_labels)
        predicted_label = classifier.test(test_instance[feature_subset])
        correct_predictions += (predicted_label == test_label)
    accuracy = correct_predictions / num_instances
    return accuracy

def greedyforwardsearch(data, labels):
    num_features = data.shape[1]
    features = set(range(num_features))
    best_accuracy = -math.inf
    best_feature_subset = set()

    classifier = Classifier()
    validator = Validator(classifier)

    selected_features = set()

    for i in range(num_features):
        max_accuracy = -math.inf
        best_feature = None
        best_candidate_features = None

        for feature in features - selected_features:
            candidate_features = selected_features | {feature}
            candidate_features_zero_based = {f for f in candidate_features}
            accuracy = validator.evaluate(data, labels, list(candidate_features_zero_based))
            print(f'Using feature(s) {candidate_features_zero_based} (1-based: {set(f+1 for f in candidate_features)}) accuracy is {accuracy*100:.2f}%')

            if (accuracy > max_accuracy) or (accuracy == max_accuracy and len(candidate_features) < len(best_candidate_features)):
                max_accuracy = accuracy
                best_feature = feature
                best_candidate_features = candidate_features

        if best_feature is not None:
            selected_features.add(best_feature)

            if max_accuracy > best_accuracy or (max_accuracy == best_accuracy and len(selected_features) < len(best_feature_subset)):
                best_accuracy = max_accuracy
                best_feature_subset = selected_features.copy()

            print(f"\nFeature set {selected_features} (1-based: {set(f+1 for f in selected_features)}) was best, accuracy is {max_accuracy*100:.2f}% \n")

    best_features_zero_based = {f for f in best_feature_subset}
    print(f"Finished search!! The best feature subset is {best_features_zero_based} (1-based: {set(f+1 for f in best_feature_subset)}), which has an accuracy of {best_accuracy*100:.2f}%")

def backwardsearch(data, labels):
    num_features = data.shape[1]
    features = set(range(num_features))
    best_accuracy = 0
    best_features = features.copy()

    classifier = Classifier()
    validator = Validator(classifier)

    while len(features) > 0:
        max_accuracy = -math.inf
        worst_feature = None
        best_candidate_features = None

        for feature in features:
            candidate_features = features - {feature}
            candidate_features_zero_based = {f for f in candidate_features}
            accuracy = validator.evaluate(data, labels, list(candidate_features_zero_based))
            print(f'Using feature(s) {candidate_features_zero_based} (1-based: {set(f+1 for f in candidate_features)}) accuracy is {accuracy*100:.2f}%')

            if (accuracy > max_accuracy) or (accuracy == max_accuracy and len(candidate_features) < len(best_candidate_features)):
                max_accuracy = accuracy
                worst_feature = feature
                best_candidate_features = candidate_features

        if max_accuracy > best_accuracy or (max_accuracy == best_accuracy and len(best_candidate_features) < len(best_features)):
            best_accuracy = max_accuracy
            best_features = best_candidate_features
        
        if worst_feature is not None:
            features.remove(worst_feature)
            print(f"\nFeature set {features} (1-based: {set(f+1 for f in features)}) was best, accuracy is {max_accuracy*100:.2f}% \n")

    best_features_zero_based = {f for f in best_features}
    print(f"Finished search!! The best feature subset is {best_features_zero_based} (1-based: {set(f+1 for f in best_features)}), which has an accuracy of {best_accuracy*100:.2f}%")

def special_algorithm(data, labels):
    num_features = data.shape[1]
    features = set(range(num_features))
    selected_features = set()
    best_accuracy = 0
    best_feature_subset = set()

    classifier = Classifier()
    validator = Validator(classifier)

    baseline_accuracy = validator.evaluate(data, labels, [])
    print(f"Baseline accuracy (no features): {baseline_accuracy*100:.2f}%")

    for i in range(num_features):
        max_accuracy = -math.inf
        best_feature = None
        best_candidate_features = None

        for feature in features - selected_features:
            candidate_features = selected_features | {feature}
            candidate_features_zero_based = {f for f in candidate_features}
            accuracy = validator.evaluate(data, labels, list(candidate_features_zero_based))
            print(f'Using feature(s) {candidate_features_zero_based} (1-based: {set(f+1 for f in candidate_features)}) accuracy is {accuracy*100:.2f}%')

            if (accuracy > max_accuracy) or (accuracy == max_accuracy and len(candidate_features) < len(best_candidate_features)):
                max_accuracy = accuracy
                best_feature = feature
                best_candidate_features = candidate_features

        if max_accuracy > best_accuracy or (max_accuracy == best_accuracy and len(selected_features | {best_feature}) < len(best_feature_subset)):
            selected_features.add(best_feature)
            best_accuracy = max_accuracy
            best_feature_subset = selected_features.copy()
            print(f"\nFeature set {selected_features} (1-based: {set(f+1 for f in selected_features)}) was best, accuracy is {best_accuracy*100:.2f}%")

            if best_accuracy > baseline_accuracy:
                print("(Improvement over baseline!)")
            else:
                print("(No improvement over baseline yet.)")

            print()  
        else:
            print("\n(Warning, Accuracy has decreased!)\n")
            break  

    selected_features_zero_based = {f for f in best_feature_subset}
    print(f"\nFinished search!! The best feature subset is {selected_features_zero_based} (1-based: {set(f+1 for f in best_feature_subset)}), which has an accuracy of {best_accuracy*100:.2f}%")


print("Welcome to Group 45 Feature Selection Algorithm.\n")

print("Type the number of the algorithm you want to run.\n")
print(" 1 = Forward Selection")
print(" 2 = Backward Elimination")
print(" 3 = Special Algorithm\n")

choice = int(input())

print(f'Using no features and "random" evaluation, I get an accuracy of {random.uniform(0,100):.2f}% \n')
print("Beginning search.\n")

if choice == 1 or choice == 2 or choice == 3:
    print("Choose the dataset to evaluate:")
    print(" 1 = Small dataset")
    print(" 2 = Large dataset")
    print(" 3 = Group 25 small dataset")
    print(" 4 = Group 25 large dataset")
    dataset_choice = int(input())

    if dataset_choice == 1:
        filename = 'data/small-test-dataset.txt'
    elif dataset_choice == 2:
        filename = 'data/large-test-dataset.txt'
    elif dataset_choice == 3:
        filename = 'data/CS170_Spring_2024_Small_data__25.txt'
    elif dataset_choice == 4:
        filename = 'data/CS170_Spring_2024_Large_data__25.txt'
    else:
        print("Invalid choice!")
        exit()

    data, labels = load_dataset(filename)

    if choice == 1:
        greedyforwardsearch(data, labels)
    elif choice == 2:
        backwardsearch(data, labels)
    elif choice == 3:
        special_algorithm(data, labels)
