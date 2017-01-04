from pandas import read_csv
from matplotlib import pyplot as plt
import seaborn

from sys import argv

def main(name):
    data = read_csv("{}.log".format(name), index_col=0)

    ax = data.plot()
    ax.set_title('Model Performance for {}\nAve. Score: {:.2f}'.format(name, data['score'].mean()))

    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')

    plt.figure()
    data.boxplot()
    plt.show()


if __name__ == '__main__':
    if len(argv) > 1:
        main(argv[1])
    else:
        print("Please provide the save_name for the model as an argument")



