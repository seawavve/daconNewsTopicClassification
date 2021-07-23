import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


def train(model,
          loss_fn,
          metric,
          optimizer,
          scheduler,
          device,
          train_dataloader,
          num_epochs,
          log_interval,
          max_grad_norm
          ):

    for e in range(num_epochs):
        train_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += metric(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                         train_acc / (batch_id + 1)))
        print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))


def test(model,
         metric,
         device,
         test_dataloader
         ):

    model.eval()
    test_eval = []
    test_acc = 0.0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += metric(out, label)

        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            test_eval.append(np.argmax(logits))

    print("test acc {}".format(test_acc / (batch_id + 1)))

    # 결과물 출력
    result = pd.DataFrame(test_eval, columns=['topic_idx'])
    result['index'] = range(45654, 45654 + len(result))
    result.set_index('index', inplace=True)
    result.to_csv('result.csv')
    pd.display(result)
