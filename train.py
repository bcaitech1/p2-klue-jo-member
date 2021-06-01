import pickle as pickle
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig
from load_data_with_ner import *
from sklearn.model_selection import train_test_split
import numpy as np
import random
import argparse

#seed 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
        
weight = torch.tensor([])
class FocalLossTrainer(Trainer) :
    def compute_loss(self, model, inputs, return_outputs=False) :
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = FocalLoss(weight=weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train(args):
    # load model and tokenizer
    seed_everything(args.seed)
    MODEL_NAME = 'xlm-roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # load dataset
    dataset = pd.read_csv('/opt/ml/input/data/train/train.tsv', delimiter='\t', header=None)
    train, dev = train_test_split(dataset, test_size=0.2, random_state=42)
    train.to_csv('/opt/ml/input/data/train/train_train.tsv', sep='\t', header=None, index=False)
    dev.to_csv('/opt/ml/input/data/train/train_dev.tsv', sep='\t', header=None, index=False)

    train_dataset = load_data('/opt/ml/input/data/train/train_train.tsv')
    dev_dataset = load_data('/opt/ml/input/data/train/train_dev.tsv')
    train_label = train_dataset['label'].values
    dev_label = dev_dataset['label'].values
    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # setting model hyperparameter
    roberta_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
    roberta_config.num_labels = 42
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME,config=roberta_config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        save_total_limit=3,  # number of total save model.
        save_steps=600,  # model saving step.
        num_train_epochs=10,  # total number of training epochs
        learning_rate=5e-6,  # learning_rate
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,
        evaluation_strategy='steps',
        eval_steps=300,
        dataloader_num_workers=4,
        label_smoothing_factor=0.5,
        logging_dir='./logs',  # directory for storing logs
        logging_steps=300
        # evaluation_strategy='steps', # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        # eval_steps = 500,            # evaluation step.
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()


def main(args):
    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
