import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from dataset import BERTDataset
from model import BERTClassifier
from runner import Trainer
from utils import calc_accuracy


def main():
    device = torch.device("cuda:0")
    bertmodel, vocab = get_pytorch_kobert_model()

    dataset_train = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/augumented_train_data.tsv", field_indices=[1, 2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/test_data.tsv", field_indices=[1, 2], num_discard_samples=1)

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    ## Setting parameters
    max_len = 64
    batch_size = 32
    warmup_ratio = 0.1
    num_epochs = 1
    max_grad_norm = 1
    log_interval = 200
    learning_rate = 5e-5

    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
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
          num_epochs=num_epochs,
          log_interval=log_interval,
          max_grad_norm=max_grad_norm
          )

    trainer.test(model=model,
         metric=calc_accuracy,
         device=device,
         test_dataloader=test_dataloader,
         )


if __name__ == "__main__":
    main()
