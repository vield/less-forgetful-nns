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

    ewc_df = pd.read_csv('ewc.csv')
    ewc1 = ewc_df[ewc_df['Group'] == 1]
    ewc2 = ewc_df[ewc_df['Group'] == 2]

    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax1.plot(group1['Epoch'], group1['TestAccuracy'])
    ax1.plot(group2['Epoch'], group2['TestAccuracy'])
    ax1.set_title('Sequential training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')

    ax2 = fig.add_subplot(312)
    ax2.plot(mixed1['Epoch'], mixed1['TestAccuracy'])
    ax2.plot(mixed2['Epoch'], mixed2['TestAccuracy'])
    ax2.set_title('Training with the full dataset')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    ax3 = fig.add_subplot(313)
    ax3.plot(ewc1['Epoch'], ewc1['TestAccuracy'])
    ax3.plot(ewc2['Epoch'], ewc2['TestAccuracy'])
    ax3.set_title('With EWC')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')

    plt.tight_layout()

    plt.show()