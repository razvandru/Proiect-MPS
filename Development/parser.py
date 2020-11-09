import sys
import pandas as pd

if __name__ == "__main__":

    read_file = pd.read_excel('mps.dataset.xlsx')
    read_file.to_csv('mps.dataset.csv', index = None, header = True)
    df = pd.DataFrame(pd.read_csv("mps.dataset.csv"))