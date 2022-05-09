import os
from typing import Tuple
import pandas as pd
from torch.utils.data import Dataset


class QadiDataset(Dataset):

    def __init__(self, data_file_name: str, qadi_data_folder: str):
        """
        Load data from specified file and put them in memory.
        Assumes the name of the column containing index/id of a tweet is "id".

        :param: data_file_name: name of file containing data
        :param: qadi_data_folder: path of folder containing the file
        """
        self.data: pd.DataFrame = pd.read_csv(
            os.path.join(qadi_data_folder, data_file_name),
            sep='\t',
            lineterminator='\n',
            index_col='id')

    def __len__(self):
        """
        :return: Amount of lines/tweets in this dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item: int) -> Tuple[str, str]:
        """
        Returns a tuple consisting of the tweet content and country label
        :param: item: index of the sample to return
        :return: sample at index item in the dataset, along with its label
        """
        text = self.data.iloc[item]["text"]
        label = self.data.iloc[item]["country_label"]

        return text, label


if __name__ == "__main__":

    data = QadiDataset("dev_set.tsv", "Data/QADI")
    print(f"There are {len(data)} samples")
    text, label = data[0]
    print(f"tweet '{text}' is from {label}")



