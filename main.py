import transformers

from QadiDataset import QadiDataset
import utils
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader


def train(train_data):

    tokenizer: transformers.PreTrainedTokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    model = AutoModel.from_pretrained("UBC-NLP/MARBERT")

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    num_epochs = 1

    # both features and labels are tuples
    for features, labels in train_loader:
        print(len(features), len(labels))


if __name__ == '__main__':

    configs = utils.read_yaml_file("config.yaml")
    dev_data = QadiDataset(configs["dev_file_name"], configs["data_files_path"])
    train(dev_data)
