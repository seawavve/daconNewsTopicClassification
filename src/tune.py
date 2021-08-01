import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import os
import pickle
import numpy as np
from argparse import ArgumentParser

from sklearn.model_selection import StratifiedShuffleSplit
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import ray
import ray.tune as tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from src.dataset import BERTDataset
from src.model import BERTClassifier
from src.utils import calc_accuracy
from src.runner import Trainer
from src.utils import set_seed


def run_hyperopt(config):
    ## Setting parameters
    max_len = 64
    batch_size = config["batch_size"]
    dropout_prob = config["dropout_prob"]
    warmup_ratio = 0.1
    num_epochs = config["num_epochs"]
    max_grad_norm = 1
    log_interval = 200
    learning_rate = config["learning_rate"]
    num_k_fold = config["num_k_fold"]
    test_size = config["test_size"]

    device = torch.device("cuda")
    bertmodel, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(bertmodel, dr_rate=dropout_prob)
    model = nn.DataParallel(model).to(device)

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    dataset = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/augumented_train_data.tsv",
                                  field_indices=[1, 2], num_discard_samples=1)
    X = [data[0] for data in dataset]
    y = [data[1] for data in dataset]
    train_val_dataloaders = []
    splitter = StratifiedShuffleSplit(n_splits=num_k_fold, test_size=test_size)
    for train_index, val_index in splitter.split(X, y):
        X_train, X_val = [X[i] for i in train_index], [X[i] for i in val_index]
        y_train, y_val = [y[i] for i in train_index], [y[i] for i in val_index]
        _train_fold = list(zip(X_train, y_train))
        _val_fold = list(zip(X_val, y_val))
        train_fold = BERTDataset(_train_fold, 0, 1, tok, max_len, True, False)
        val_fold = BERTDataset(_val_fold, 0, 1, tok, max_len, True, False)
        train_fold_dataloader = torch.utils.data.DataLoader(train_fold, batch_size=batch_size, num_workers=5)
        val_fold_dataloader = torch.utils.data.DataLoader(val_fold, batch_size=batch_size, num_workers=5)
        train_val_dataloaders += [[train_fold_dataloader, val_fold_dataloader]]

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    t_total = len(train_val_dataloaders[0][0]) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    trainer = Trainer()

    accuracy = []
    for train_fold_dataloader, val_fold_dataloader in train_val_dataloaders:
        trainer.train(model=model,
                      loss_fn=loss_fn,
                      metric=calc_accuracy,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      device=device,
                      train_dataloader=train_fold_dataloader,
                      num_epochs=num_epochs,
                      log_interval=log_interval,
                      max_grad_norm=max_grad_norm
                      )
        trainer.test(model=model,
                     metric=calc_accuracy,
                     device=device,
                     test_dataloader=val_fold_dataloader
                     )
        accuracy += [trainer.test_acc]

    return {"accuracy": np.mean(accuracy)}


def main(args):
    # set hyperparameters which will not be tuned
    seed = 0
    metric_name = "accuracy"

    set_seed(seed, 1)
    ray.init()

    # initialize search space
    space = {
        "num_epochs": args.num_epochs,
        "batch_size": tune.choice([8, 16, 32, 64, 128]),
        "learning_rate": tune.uniform(1e-5, 1e-4),
        "dropout_prob": tune.uniform(0.0, 0.5),
        "num_k_fold": args.num_k_fold,
        "test_size": args.test_size,
        "num_gpus": len(args.gpus.split(","))
    }

    # can add additional parameters to tune
    # space["max_len"] = tune.choice([64,128])
    # space["warmup_ratio"] = tune.uniform(0.0,0.2)
    # space["max_grad_norm"] = tune.choice([1])

    hyperopt_search = HyperOptSearch(metric=metric_name, mode="max")

    name = f"ne{args.num_epochs}_nk{args.num_k_fold}_ts{args.test_size}_ns{args.num_search}_{args.exp_id}"
    path = f"/home/junhyun/log/dacon_news/hyperopt/{name}"
    analysis = tune.run(run_hyperopt,  # (callable) experiment or list of experiments
                        config=space,
                        search_alg=hyperopt_search,
                        num_samples=args.num_search,
                        resources_per_trial={'gpu': space["num_gpus"]},
                        log_to_file=True,
                        local_dir=path,
                        )

    best_trial = analysis.get_best_trial(metric=metric_name, mode="max", scope="all")

    all_dataframes = analysis.trial_dataframes
    all_configs = analysis.get_all_configs()
    keys = list(all_dataframes.keys())
    found_augment_policy_configs = []
    for idx, key in enumerate(keys):
        found_augment_policy_config = all_configs[keys[idx]]
        score = all_dataframes[key][metric_name].values[0]
        found_augment_policy_config[metric_name] = score
        found_augment_policy_configs.append(found_augment_policy_config)
    results = sorted(found_augment_policy_configs, key=lambda x: x[metric_name], reverse=True)
    print("Best trial config: {}".format(best_trial.config))
    os.makedirs(f"/home/junhyun/log/dacon_news/tune/ne{args.num_epochs}_nk{args.num_k_fold}_ts{args.test_size}_ns{args.num_search}_{args.exp_id}", exist_ok=True)
    pickle.dump(results, open(os.path.join(f"/home/junhyun/log/dacon_news/tune/ne{args.num_epochs}_nk{args.num_k_fold}_ts{args.test_size}_ns{args.num_search}_{args.exp_id}", "result.pkl"), "wb"))
    ray.shutdown()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", "-ne", type=int, default=1)
    parser.add_argument("--num_k_fold", "-nk", type=int, default=5)
    parser.add_argument("--test_size", "-ts", type=float, default=0.2)
    parser.add_argument("--num_search", "-ns", type=int, default=20)
    parser.add_argument("--exp_id", "-e", type=str, default="first_trial")
    parser.add_argument("--num_thread", "-nt", type=int, default=1)
    parser.add_argument("--gpus", "-g", type=str, default="4,5,6,7")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # os.environ["CUDA_LAUNCH_BLOCKING"] = str(args.num_thread)
    main(args)
    # python -m src.tune -ne 1 -nk 5 -ts 0.1 -e first_trial -nt 1 -g 0

    space = {
        "num_epochs": args.num_epochs,
        "batch_size": 32,
        "learning_rate": 1e-5,
        "dropout_prob": 0.1,
        "num_k_fold": args.num_k_fold,
        "test_size": args.test_size,
        "num_gpus": len(args.gpus.split(","))
    }
    # run_hyperopt(space)