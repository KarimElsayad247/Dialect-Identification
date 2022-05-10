import os
from typing import Tuple, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class QadiDataset(Dataset):

    def __init__(self, data_file_name: str, qadi_data_folder: str, tokenizer: PreTrainedTokenizer):
        """
        Load data from specified file and put them in memory.
        Assumes the name of the column containing index/id of a tweet is "id".

        :param: data_file_name: name of file containing data
        :param: qadi_data_folder: path of folder containing the file
        :param: tokenizer: huggingface AutoTokenizer to use in tokenizing dataset text
        """
        self.data: pd.DataFrame = pd.read_csv(
            os.path.join(qadi_data_folder, data_file_name),
            sep='\t',
            lineterminator='\n',
            index_col='id')
        self.encodings = tokenizer(self.data["text"].tolist(), padding=True)
        self.labels = self.data["country_label"]
        self.label_nums = {country: i for i, country in enumerate(self.labels.unique())}

    def __len__(self):
        """
        :return: Amount of lines/tweets in this dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        """
        :param: idx: index of the sample to return
        :return: encoding of the tweet, and its label
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_nums[self.labels[idx]])
        return item


if __name__ == "__main__":

    import transformers

    data = QadiDataset("dev_set.tsv", "Data/QADI", transformers.AutoTokenizer.from_pretrained("UBC-NLP/MARBERT"))
    print(f"There are {len(data)} samples")
    sample = data[0]
    print(sample)



