from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


Data = namedtuple("Data", ["filename", "title"])


if __name__ == "__main__":

    # Use seaborn default styles
    sns.set()

    # The main.py script automatically logs results in these files
    # depending on the mode used
    data = [
        Data("simple.csv", "Sequential training"),
        Data("mixed.csv", "Training with the full dataset"),
        Data("l2.csv", "With a uniform quadratic penalty"),
        Data("ewc.csv", "With EWC")
    ]

    fig = plt.figure(figsize=(15,12))

    total_plots = len(data)
    axes = []
    dfs = []

    # Stacked plots that show how accuracy on each trained group changes
    # as training progresses.
    # Data from both training sessions is included in the case of
    # sequential training (it will be obvious from the plot when we started
    # training on the second dataset).
    for i in range(len(data)):
        axes.append(fig.add_subplot(total_plots, 1, i+1))
        ax = axes[-1]

        dfs.append(pd.read_csv(data[i].filename))
        df = dfs[-1]

        group_numbers = sorted(df['Group'].unique())

        for num in group_numbers:
            group = df[df['Group'] == num]
            ax.plot(group['Epoch'], group['TestAccuracy'])

        ax.set_title(data[i].title)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy')

        # Format accuracy as percentages ... as it turns out, that's a built-in
        # now in the new formatting mini language
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Prevent subfigure labels etc from being covered by other things
    plt.tight_layout()

    plt.show()