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
from src.model import BERTClassifier
from src.runner import Trainer
from src.utils import calc_accuracy


def main(args):
    device = torch.device("cuda:0")
    bertmodel, vocab = get_pytorch_kobert_model()

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    ## Setting parameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    num_epochs = args.num_epochs

    max_len = 64
    warmup_ratio = 0.1
    max_grad_norm = 1
    log_interval = 200

    dataset_train = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/augumented_train_data.tsv",
                                        field_indices=[1, 2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/test_data.tsv",
                                       field_indices=[1, 2], num_discard_samples=1)
    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

    model = BERTClassifier(bertmodel, dr_rate=dropout_rate).to(device)

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
                  test_dataloader=test_dataloader,
                  num_epochs=num_epochs,
                  log_interval=log_interval,
                  max_grad_norm=max_grad_norm,
                  expid=args.expid
                  )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.5)
    parser.add_argument("--num_epochs", "-ne", type=int, default=1)
    parser.add_argument("--expid", type=str, default="test")
    args = parser.parse_args()
    args.expid = f"bs{args.batch_size}_lr{args.learning_rate}_dr{args.dropout_rate}_ne{args.num_epochs}"
    main(args)
