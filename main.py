import gc

import numpy as np
import torch
import transformers

from QadiDataset import QadiDataset
import utils
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_metric, Metric
from torch.utils.data import DataLoader



accuracy_metric = load_metric("accuracy")


# see https://huggingface.co/docs/transformers/training#metrics
def compute_metric_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


# see https://huggingface.co/docs/transformers/training#train
def train(configs):

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cpu')

    tokenizer: transformers.PreTrainedTokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERT", num_labels=18)
    model.to(device)

    dev_data = QadiDataset(configs["dev_file_name"], configs["data_files_path"], tokenizer)
    train_data = QadiDataset(configs["dev_file_name"], configs["data_files_path"], tokenizer)


    # train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    num_epochs = 1

    training_arguments: TrainingArguments = TrainingArguments(
        output_dir="test_trainer",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size per device during evaluation
        evaluation_strategy=transformers.IntervalStrategy.EPOCH)


    trainer: Trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_data,
        eval_dataset=dev_data,
        compute_metrics=compute_metric_accuracy
    )

    trainer.train()
    # # both features and labels are tuples
    # for features, labels in train_loader:
    #     print(len(features), len(labels))


if __name__ == '__main__':

    configs = utils.read_yaml_file("config.yaml")
    train(configs)
