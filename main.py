from Trees import *
from Nodes import *
from Evaluation_functions import *

if __name__ == '__main__':
    # Initialise random seed
    seed = 6345789
    rg = default_rng(seed)

    # Load datasets
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')

    # Make test and train sets for the clean and noisy datasets
    noisy_train, noisy_test = train_test_split(noisy_data, test_proportion=0.1, random_generator=rg)
    clean_train, clean_test = train_test_split(clean_data, test_proportion=0.1, random_generator=rg)

    # Make a decision tree for the whole clean and noisy datasets.
    clean_tree = decision_tree_learning(clean_train)
    noisy_tree = decision_tree_learning(noisy_train)
    print(clean_tree)
    print(noisy_tree)

    # Below evaluate our tree with classification metrics over the clean and noisy datasets

    # 10-fold cross validation on clean data set
    clean_cv = get_cv_metrics(clean_data, rg)
    for key, value in clean_cv.items():
        print(key)
        print(value)

    # 10-fold cross validation on noisy data set
    noisy_cv = get_cv_metrics(noisy_data, rg)
    for key, value in noisy_cv.items():
        print(key)
        print(value)

    # Nested cross validation on pruned trees on the clean data set
    clean_cv = get_nested_cv_metrics(clean_data, rg)
    for key, value in clean_cv.items():
        print(key)
        print(value)

    # Nested cross validation on pruned trees on the clean noisy data set
    noisy_cv = get_nested_cv_metrics(noisy_data, rg)
    for key, value in noisy_cv.items():
        print(key)
        print(value)

