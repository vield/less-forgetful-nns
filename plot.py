from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


Data = namedtuple("Data", ["filename", "title"])


if __name__ == "__main__":
    sns.set()

    data = [
        Data("simple.csv", "Sequential training"),
        Data("mixed.csv", "Training with the full dataset"),
        Data("ewc.csv", "With EWC")
    ]

    fig = plt.figure()

    total_plots = len(data)
    axes = []
    dfs = []

    for i in range(len(data)):
        axes.append(fig.add_subplot(total_plots, 1, i+1))
        ax = axes[-1]

        dfs.append(pd.read_csv(data[i].filename))
        df = dfs[-1]
        group1 = df[df['Group'] == 1]
        group2 = df[df['Group'] == 2]

        ax.plot(group1['Epoch'], group1['TestAccuracy'])
        ax.plot(group2['Epoch'], group2['TestAccuracy'])

        ax.set_title(data[i].title)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy')

        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    plt.tight_layout()

    plt.show()