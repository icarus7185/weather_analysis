import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_histogram_single(data_arr):

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot histogram with count on the first y-axis
    sns.histplot(data_arr, ax=ax1, stat="count", color="skyblue")
    ax1.set_ylabel("Count")

    # Create a second y-axis for density
    ax2 = ax1.twinx()
    sns.histplot(data_arr, ax=ax2, stat="density", kde=True, line_kws={'linewidth': 2})
    ax2.set_ylabel("Density")
    ax2.grid(False)

    # plt.title(feature)
    plt.show()

def visualize_histogram_full(dataframe):
    row_cnt = dataframe.columns.size // 3 + 1
    for i in range(row_cnt):
        #plt.figure(figsize=(16, 5))
        fig, (AX1, AX2, AX3) = plt.subplots(1, 3, figsize=(18, 5))
        AXGRID = [AX1, AX2, AX3]

        for j in range(3):
            idx = i * 3 + j
            if idx < dataframe.columns.size:
                #plt.subplot(1, 3, j+1).set_title(data.columns[idx])
                
                # Create a figure
                #fig, ax1 = AXGRID[j].subplots(figsize=(8, 5))

                # Plot histogram with count on the first y-axis
                sns.histplot(dataframe[dataframe.columns[idx]], ax=AXGRID[j], stat="count", color="skyblue")
                AXGRID[j].set_ylabel("Count")

                # Create a second y-axis for density
                ax2 = AXGRID[j].twinx()
                sns.histplot(dataframe[dataframe.columns[idx]], ax=ax2, stat="density", kde=True, line_kws={'linewidth': 2})
                ax2.set_ylabel("Density")
                ax2.grid(False)

        plt.tight_layout()
        plt.show()

def visualize_histogram_crop(data_arr, feature_name, min_cnt, max_cnt):
    plt.figure(figsize=(16, 5))
 
    counts, bin_edges = np.histogram(data_arr, bins=50, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #derivative_counts = np.gradient(counts, bin_centers[1] - bin_centers[0])

    plt.subplot(1, 2, 1).set_title(f"{feature_name} full")
    plt.bar(bin_centers, counts, width=bin_centers[1] - bin_centers[0], edgecolor='black')

    plt.subplot(1, 2, 2).set_title(f"{feature_name} zoom in")
    plt.ylim(min_cnt, max_cnt)
    plt.bar(bin_centers, counts, width=bin_centers[1] - bin_centers[0], edgecolor='black')

    plt.show()

def visualize_each_feature_in_1D_vertical(dataframe):
    row_cnt = dataframe.columns.size // 6 + 1
    dumyx = np.full(len(dataframe[dataframe.columns[0]]), 0)

    # Create a figure
    plt.figure(figsize=(16, row_cnt*3))

    for col in range(0, dataframe.columns.size):
        plt.subplot(row_cnt, 6, col+1).set_title(dataframe.columns[col])
        plt.scatter(dumyx, dataframe.iloc[:,col])
        plt.tight_layout()

    plt.show()

def display_confusion_matrix(Clf, varian_group_name, y_label, save_name=None):
    varian_size = len(Clf)
    fig, axs = plt.subplots(1, varian_size, figsize=(18, 5))

    for i in range(varian_size):
        Clf[i].confusion_matrix(axs[i], y_label)
        axs[i].set_title(varian_group_name[i])
        axs[i].set_xlabel('Predicted Label')
        axs[i].set_ylabel('True Label')

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(f'{save_name}_confusion_matrix.png')

    # Display the figure with subplots
    plt.show()