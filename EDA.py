import numpy as np
import matplotlib.pyplot as plt


def plot_hist(data):
    plt.figure()
    figure, axis = plt.subplots(2, 2)
    axes = axis.flat
    count = 0
    for label in np.unique(data[:, -1]):
        # Plot a distribution for each label
        data_slice = data[data[:, -1] == label]
        axes[count].hist(data_slice[:, :-1])
        axes[count].set_title(f'Signals for room {label}')
        count += 1
    plt.tight_layout()
    labels = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    plt.figlegend(labels, loc='lower center', ncol=7, labelspacing=0.)
    plt.show()


if __name__ == '__main__':

    # Load the data sets

    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    print(clean_data[:5, :])
    print(noisy_data[:5, :])

    # Confirm that the clean and noisy datasets have the same number of features (columns)
    assert clean_data.shape[1] == noisy_data.shape[1]

    # Confirm that both the clean and noisy datasets have the same number of classes
    assert np.all(np.unique(clean_data[:, -1]).sort() == np.unique(noisy_data[:, -1]).sort())

    for k in range(noisy_data.shape[1] - 1):
        print(f'Column {k}')
        print(f'clean max is {np.max(clean_data[:, k])}')
        print(f'noisy max is {np.max(noisy_data[:, k])}')
        print(f'clean min is {np.min(clean_data[:, k])}')
        print(f'noisy min is {np.min(noisy_data[:, k])}')
        print(f'clean mean is {np.mean(clean_data[:, k])}')
        print(f'noisy mean is {np.mean(noisy_data[:, k])}')

    print([(sum(clean_data[:, -1] == x) / clean_data.shape[0]) for x in np.unique(clean_data[:, -1])])
    print([(sum(noisy_data[:, -1] == x) / noisy_data.shape[0]) for x in np.unique(noisy_data[:, -1])])

    # Plot Histograms for both datasets
    plot_hist(clean_data)
    plot_hist(noisy_data)
