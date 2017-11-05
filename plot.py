import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



if __name__ == "__main__":
    sns.set()

    simple_df = pd.read_csv('simple.csv')
    group1 = simple_df[simple_df['Group'] == 1]
    group2 = simple_df[simple_df['Group'] == 2]

    mixed_df = pd.read_csv('mixed.csv')
    mixed1 = mixed_df[mixed_df['Group'] == 1]
    mixed2 = mixed_df[mixed_df['Group'] == 2]

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(group1['Epoch'], group1['TestAccuracy'])
    ax1.plot(group2['Epoch'], group2['TestAccuracy'])
    ax1.set_title('Sequential training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')

    ax2 = fig.add_subplot(212)
    ax2.plot(mixed1['Epoch'], mixed1['TestAccuracy'])
    ax2.plot(mixed2['Epoch'], mixed2['TestAccuracy'])
    ax2.set_title('Training with the full dataset')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()

    plt.show()