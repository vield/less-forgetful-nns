import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



if __name__ == "__main__":
    sns.set()

    simple_df = pd.read_csv('simple.csv')
    group1 = simple_df[simple_df['Group'] == 1]
    group2 = simple_df[simple_df['Group'] == 2]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(group1['Epoch'], group1['TestAccuracy'])
    ax1.plot(group2['Epoch'], group2['TestAccuracy'])
    ax1.set_title('Sequential training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')

    plt.show()