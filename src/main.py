import json
import os
import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
from argparse import ArgumentParser

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from src.dataset import BERTDataset
from src.model import BERTClassifier, BERTEnsemble
from src.runner import Trainer
from src.utils import calc_accuracy


def main(args):
    device = torch.device("cuda:0")
    max_len = 64
    warmup_ratio = 0.1
    max_grad_norm = 1
    log_interval = 200

    # Setting parameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    num_epochs = args.num_epochs

    bertmodel, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(bertmodel, dr_rate=dropout_rate).to(device)
    loss_fn = nn.CrossEntropyLoss()

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    dataset_train = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/augumented_train_data.tsv",
                                        field_indices=[1, 2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/test_data.tsv",
                                       field_indices=[1, 2], num_discard_samples=1)
    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    trainer = Trainer()
    trainer.train(model=model,
                  loss_fn=loss_fn,
                  metric=calc_accuracy,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  device=device,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  num_epochs=num_epochs,
                  log_interval=log_interval,
                  max_grad_norm=max_grad_norm,
                  expid=args.expid,
                  )


def bagging(args):
    # Setting parameters
    device = torch.device("cuda:0")
    max_len = 64
    warmup_ratio = 0.1
    max_grad_norm = 1
    log_interval = 200
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    num_epochs = args.num_epochs

    bertmodel, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(bertmodel, dr_rate=dropout_rate).to(device)
    loss_fn = nn.CrossEntropyLoss()

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    dataset_train = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/augumented_train_data.tsv",
                                        field_indices=[1, 2], num_discard_samples=1)
    bootstrap_train_dataloaders = []  # todo
    for _ in range(5):
        data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
        train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
        bootstrap_train_dataloaders += (train_dataloader,)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total
    )

    trainer = Trainer()
    for idx, train_dataloader in bootstrap_train_dataloaders:
        expid = args.expid + f"_B{idx}"
        trainer.train(model=model,
                      loss_fn=loss_fn,
                      metric=calc_accuracy,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      device=device,
                      train_dataloader=train_dataloader,
                      test_dataloader=None,
                      num_epochs=num_epochs,
                      log_interval=log_interval,
                      max_grad_norm=max_grad_norm,
                      expid=expid,
                      )


def ensemble(args):
    device = torch.device("cuda:0")
    max_len = 64
    batch_size = 64

    bertmodel, vocab = get_pytorch_kobert_model()
    assert args.ckpt, "please specify path of json file that contains ckpt file paths."
    ckpt_list = json.load(open(args.ckpt))["ckpt_list"]
    model = BERTEnsemble(bertmodel, ckpt_list, device).to(device)

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    dataset_test = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/test_data.tsv",
                                       field_indices=[1, 2], num_discard_samples=1)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

    trainer = Trainer()
    trainer.test(model=model,
                 metric=calc_accuracy,
                 device=device,
                 test_dataloader=test_dataloader,
                 epoch=0,
                 expid=args.expid
                 )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = ArgumentParser()
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.5)
    parser.add_argument("--num_epochs", "-ne", type=int, default=1)
    parser.add_argument("--ensemble", "-e", action="store_true")
    parser.add_argument("--ckpt", type=str, default="/home/junhyun/projects/dacon_news/ckpt_paths.json")
    args = parser.parse_args()

    if args.ensemble:
        args.expid = "ensemble"
        ensemble(args)
    else:
        args.expid = f"bs{args.batch_size}_lr{args.learning_rate}_dr{args.dropout_rate}"
        main(args)
