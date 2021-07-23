import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import os
import pickle
from argparse import ArgumentParser

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import ray
import ray.tune as tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from dataset import BERTDataset
from model import BERTClassifier
from utils import calc_accuracy
from runner import Trainer
from utils import set_seed


def run_hyperopt(config):
    device = torch.device("cuda:0")
    bertmodel, vocab = get_pytorch_kobert_model()

    dataset_train = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/augumented_train_data.tsv",
                                        field_indices=[1, 2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset("/home/junhyun/projects/dacon_news/data/test_data.tsv", field_indices=[1, 2],
                                       num_discard_samples=1)

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    ## Setting parameters
    max_len = 64
    batch_size = config["batch_size"]
    warmup_ratio = 0.1
    num_epochs = config["num_epochs"]
    max_grad_norm = 1
    log_interval = 200
    learning_rate = config["learning_rate"]

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

    return {"accuracy": trainer.test_acc}


def main(args):
    # set hyperparameters which will not be tuned
    seed = 0
    num_epochs = 1
    num_search = 5  # for real tuning, use 50
    metric_name = "accuracy"
    num_gpus = 1  # number of gpus to be used

    set_seed(seed, 1)
    ray.init()

    # initialize search space
    space = {
        "num_epochs": num_epochs,
        "batch_size": tune.choice([8, 12, 16, 32, 64]),
        "learning_rate": tune.uniform(1e-5, 1e-4),
        # "dropout_prob": tune.choice([0.0, 0.1, 0.2, 0.3]),
    }

    # can add additional parameters to tune
    # space["max_len"] = tune.choice([64,128])
    # space["warmup_ratio"] = tune.uniform(0.0,0.2)
    # space["max_grad_norm"] = tune.choice([1])

    hyperopt_search = HyperOptSearch(space, metric=metric_name, mode="max")

    analysis = tune.run(run_hyperopt,  # (callable) experiment or list of experiments
                        search_alg=hyperopt_search,
                        num_samples=num_search,
                        resources_per_trial={'gpu': num_gpus},
                        name=args.exp_id,
                        log_to_file=True,
                        local_dir=f"/home/junhyun/projects/dacon_news/log/tune/"
                        # trial_dirname_creator=trial_dirname_creator
                        )

    best_trial = analysis.get_best_trial(metric_name, "max")

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
    os.makedirs(f"/home/junhyun/log/dacon_news/tune/{args.exp_id}", exist_ok=True)
    pickle.dump(results, open(os.path.join(f"/home/junhyun/log/dacon_news/tune/{args.exp_id}", "result.pkl"), "wb"))
    ray.shutdown()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_id", "-e", type=str, default="none")
    args = parser.parse_args()
    main(args)
