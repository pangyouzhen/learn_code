import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def main():
    sns.set_styple("whitegrid", {'font.sans-serif': ['simhei', 'Arial']})
    offline = pd.read_excel("./offline_case.xlsx")
    vc = offline["match"].value_counts()
    sns.barplot(x=vc.index, y=vc.values)
    plt.show()


if __name__ == '__main__':
    main()
