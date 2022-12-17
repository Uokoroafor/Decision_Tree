import numpy as np
from numpy.random import default_rng
from copy import deepcopy
from Trees import *


def train_test_split(data, test_proportion=0.1, random_generator=default_rng()):
    """ Splits dataset into training and test sets, according to the given
            test set proportion.

        Args:
            data (np array): numpy array with shape (N,K) that hosts the data, the last column is the labels
            test_proportion (float): the desired proportion of data to test
            random_generator (np.random.Generator): A pre-seeded random generator

        Returns:
            train_data (np.ndarray): Training instances shape (N_train, K)
            test_data (np.ndarray): Test instances shape (N_test, K)
        """
    shuffled_indices = random_generator.permutation(len(data))

    test_end = round((len(shuffled_indices) * test_proportion))
    test_data = data[shuffled_indices[:test_end], :]
    train_data = data[shuffled_indices[test_end:], :]

    classes = np.unique(data[:, -1])
    test_classes = np.unique(test_data[:, -1])
    # Make sure all classes are featured in the test set
    assert len(classes) == len(
        test_classes), "Not all classes were featured in the test_set. Perhaps try a different random seed?"
    return train_data, test_data


def cross_validation_split(data, index, folds=10):
    """ Selects the indexed slice from a shuffled dataset returning the slice as test set and the rest as the train set for 10 fold cross validation.

        Args:
            data (np array): numpy array with shape (N,K) that hosts the data, the last column is the labels
            index (integer): an integer from 0 to number of folds-1
            folds (integer): the number of folds for cross validation

        Returns:
            train_data (np.ndarray): Training instances shape (N_train, K)
            test_data (np.ndarray): Test instances shape (N_test, K)

        """
    assert type(index) == int and (0 <= index <= folds), "index should be integer in [0,9]"
    index_start = round((len(data) / folds) * index)
    index_end = round((len(data) / folds) * (index + 1))
    test_data = data[index_start:index_end]
    train_data = np.delete(data, slice(index_start, index_end), 0)
    return train_data, test_data


def nested_validation_split(data, validation_index, test_index, folds=10):
    """ Selects the indexed slice from a shuffled dataset returning the slices as validation and test sets and rest as the train set for 10 fold cross validation.

        Args:
            data (np array): numpy array with shape (N,K) that hosts the data, the last column is the labels
            validation_index (integer): an integer from 0 to folds
            test_index (integer): an integer from 0 to folds
            folds (integer): the number of folds for cross validation

        Returns:
            train_data (np array): Training instances shape (N_train, K)
            validation_data (np array): Validation instances shape (N_validation, K)
            test_data (np array): Test instances shape (N_test, K)
        """
    assert test_index != validation_index, "Validation and test sets are the same"

    # Test indices
    index1_start = round((len(data) / folds) * test_index)
    index1_end = round((len(data) / folds) * (test_index + 1))

    # Validation indices
    index2_start = round((len(data) / folds) * validation_index)
    index2_end = round((len(data) / folds) * (validation_index + 1))

    test_data = data[index1_start:index1_end]
    validation_data = data[index2_start:index2_end]

    # The remainder is the train data
    # Remove the higher index first
    if index1_end > index2_end:
        # First delete the test then the validation
        train_data = np.delete(data, slice(index1_start, index1_end), 0)
        train_data = np.delete(train_data, slice(index2_start, index2_end), 0)
    else:
        # First delete the validation then the test
        train_data = np.delete(data, slice(index2_start, index2_end), 0)
        train_data = np.delete(train_data, slice(index1_start, index1_end), 0)

    return train_data, validation_data, test_data


def get_accuracy(data, tree):
    predictions = tree.classify_tests(data)
    counter = np.sum(predictions == data[:, -1])

    return counter / len(data)


def get_confusion_matrix(data, tree):
    """Assumes the target variable is in the last column"""

    # matrix = np.zeros((4, 4))
    classes = len(np.unique(data[:, -1]))
    matrix = np.zeros((classes, classes))

    for item in data:
        true_label = int(item[-1]) - 1
        predicted_label = int(tree.classify_example(item)) - 1

        matrix[true_label][predicted_label] += 1

    return matrix


def accuracy_average(dataset, random_generator=default_rng(), max_depth=10000, folds=10):
    accuracy_total = 0
    data = random_generator.permutation(dataset)  # Shuffle the data for cross validation
    for i in range(folds):
        print(f"Accuracy for fold {i + 1}")
        train_data, test_data = cross_validation_split(data, i)
        tree = decision_tree_learning(train_data, max_depth)
        print(get_accuracy(test_data, tree))
        print('############################')
        accuracy_total += get_accuracy(test_data, tree)

    accuracy = accuracy_total / folds

    return accuracy


def confusion_matrix_total(dataset, random_generator=default_rng(), prints=False, max_depth=10000):
    classes = len(np.unique(dataset[:, -1]))
    matrix = np.zeros((classes, classes))
    data = random_generator.permutation(dataset)
    for i in range(10):
        train_data, test_data = cross_validation_split(data, i)
        tree = decision_tree_learning(train_data, max_depth)
        if prints:
            print(f'Confusion matrix for fold {i + 1}')
            print(get_confusion_matrix(test_data, tree))
            print('############################')
        matrix += get_confusion_matrix(test_data, tree)

    return matrix / 10


def precision_per_class(matrix):
    precision = np.zeros(matrix.shape[0])

    for i in range(len(matrix[0])):
        if np.sum(matrix[:, i]) == 0:
            precision[i] = 0
        else:
            precision[i] = matrix[i][i] / np.sum(matrix[:, i])

    return precision


def recall_per_class(matrix):
    recall = np.zeros(matrix.shape[0])

    for i in range(len(matrix[0])):
        if np.sum(matrix[i, :]) == 0:
            recall[i] = 0
        else:
            recall[i] = matrix[i][i] / np.sum(matrix[i, :])

    return recall


def f1_per_class(precision, recall):
    # To avoid errors, we set recall to 1 in cases where precision and recall sum to zero
    # This ensures we get an f1 of zero in this case and not an error
    recall[(recall == 0) & (precision == 0)] = 1
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_cv_metrics(dataset, random_generator=default_rng(), max_depth=10000, folds=10):
    """ Aggregates accuracy, confusion matrices, precision, recalls and f1s for 10-fold cross validation

        Args:
            dataset (np array): numpy array with shape (N,K) that hosts the data, the last column is the labels
            random_generator (np.random.Generator): A pre-seeded random generator
            max_depth (integer): Maximum depth of the trees being trained to 10,000

        Returns:
            dict :
                total_confusion_matrix (average of the confusion matrices of each fold),
                Total_accuracy (average of the total accuracy of each fold),
                precisions (average of precisions of each fold),
                recalls (average of recalls of each fold),
                f1s (average of f1s of each fold)
        """

    classes = len(np.unique(dataset[:, -1]))
    matrix = np.zeros((classes, classes))
    accuracy_total = 0
    data = random_generator.permutation(dataset)
    recalls = np.zeros(classes)
    precisions = np.zeros(classes)
    f1s = np.zeros(classes)
    for i in range(folds):
        train_data, test_data = cross_validation_split(data, i)
        tree = decision_tree_learning(train_data, max_depth)
        fold_matrix = get_confusion_matrix(test_data, tree)
        fold_accuracy = get_accuracy(test_data, tree)
        matrix += fold_matrix
        accuracy_total += fold_accuracy
        precision = precision_per_class(fold_matrix)
        recall = recall_per_class(fold_matrix)
        class_f1 = f1_per_class(precision, recall)
        recalls += recall
        precisions += precision
        f1s += class_f1
    return {'total_confusion_matrix': matrix / folds, 'Total_accuracy': accuracy_total / folds,
            'precisions': precisions / folds,
            'recalls': recalls / folds, 'f1s': f1s / folds}


def prune_tree(test_data, tree):
    new_tree = deepcopy(tree)
    # x, y = len(new_tree.nodes), len(new_tree.leaves)
    count = 0
    # Want the loop to continue until no further nodes can be removed
    nodes_stable = False
    nodes = [node for node in new_tree.nodes]
    while not nodes_stable:
        node_count = len(nodes)
        for node in nodes:
            count += 1
            main_error = new_tree.get_errors(test_data)
            if node.has_leaves():
                new_tree.make_leaf(node)
                new_error = new_tree.get_errors(test_data)
                new_tree.make_branch(node)
                if new_error <= main_error:
                    new_tree.prune_node(node)
                    nodes.remove(node)
        if node_count == len(nodes):
            nodes_stable = True

    return new_tree


def get_nested_cv_metrics(dataset, random_generator=default_rng(), max_depth=10000, folds=10):
    """ Aggregates accuracy, confusion matrices, precision, recalls and f1s for the nested cross validation

        Args:
            dataset (np array): numpy array with shape (N,K) that hosts the data, the last column is the labels
            random_generator (np.random.Generator): A pre-seeded random generator
            max_depth (integer): Maximum depth of the trees being trained to 10,000
            folds (integer): Number of folds

        Returns:
            dict :
                total_confusion_matrix (average of the confusion matrices of each fold),
                Total_accuracy (average of the total accuracy of each fold),
                precisions (average of precisions of each fold),
                recalls (average of recalls of each fold),
                f1s (average of f1s of each fold),
                average_depth (average depth of trained trees of each fold before pruning),
                average_pruned_depth (average depth of pruned trees after pruning)

        """
    classes = len(np.unique(dataset[:, -1]))
    pruned_accuracy_total = 0
    pruned_matrix = np.zeros((classes, classes))
    data = random_generator.permutation(dataset)  # Shuffle the original to randomise it
    pruned_recalls = np.zeros(classes)
    precisions, pruned_precisions = np.zeros(classes), np.zeros(classes)
    f1s, pruned_f1s = np.zeros(classes), np.zeros(classes)
    average_depth = 0
    average_pruned_depth = 0
    # 90 Trees
    count = folds * (folds - 1)
    for i in range(folds):
        for j in range(folds):
            if i != j:
                train_data, validation_data, test_data = nested_validation_split(data, i, j)
                tree = decision_tree_learning(train_data, max_depth)
                average_depth += tree.get_max_depth()

                pruned_tree = prune_tree(validation_data, tree)
                average_pruned_depth += pruned_tree.get_max_depth()

                pruned_fold_matrix = get_confusion_matrix(test_data, pruned_tree)
                pruned_fold_accuracy = get_accuracy(test_data, pruned_tree)
                pruned_matrix += pruned_fold_matrix
                pruned_accuracy_total += pruned_fold_accuracy
                pruned_precision = precision_per_class(pruned_fold_matrix)
                pruned_recall = recall_per_class(pruned_fold_matrix)
                pruned_class_f1 = f1_per_class(pruned_precision, pruned_recall)
                pruned_recalls += pruned_recall
                pruned_precisions += pruned_precision
                pruned_f1s += pruned_class_f1

    return dict(total_confusion_matrix=pruned_matrix / count, Total_accuracy=pruned_accuracy_total / count,
                precisions=pruned_precisions / count, recalls=pruned_recalls / count, f1s=pruned_f1s / count,
                average_depth=average_depth / count, average_pruned_depth=average_pruned_depth / count)


if __name__ == '__main__':
    pass
