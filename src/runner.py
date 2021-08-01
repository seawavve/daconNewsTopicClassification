import torch
import pandas as pd
import numpy as np
from tqdm import tqdm



class Trainer:
    def __init__(self):
        pass

    def train(self,
              model,
              loss_fn,
              metric,
              optimizer,
              scheduler,
              device,
              train_dataloader,
              test_dataloader,
              num_epochs,
              log_interval,
              max_grad_norm,
              expid
              ):
        for epoch in range(num_epochs):
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
                    print("epoch {} batch id {} loss {} train acc {}".format(epoch + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                             train_acc / (batch_id + 1)))
            print("epoch {} train acc {}".format(epoch + 1, train_acc / (batch_id + 1)))

            # self.test(model, metric, device, test_dataloader, epoch, expid)
            torch.save(model.state_dict(), f"/home/junhyun/projects/dacon_news/ckpt/{expid}_epoch{epoch + 1}.tar")

    def test(self,
             model,
             metric,
             device,
             test_dataloader,
             epoch,
             expid,
             mcdrop=False
             ):
        model.train() if mcdrop else model.eval()
        test_eval = []
        test_acc = 0.0
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            if mcdrop:
                assert isinstance(mcdrop, int), "mcdrop must be boolean value False or positive integer value"
                out = 0
                for i in range(mcdrop):
                    out += model(token_ids, valid_length, segment_ids) / mcdrop
            else:
                out = model(token_ids, valid_length, segment_ids)
            test_acc += metric(out, label)

            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()
                test_eval.append(np.argmax(logits))

        print(f"test acc at epoch {epoch}: {test_acc / (batch_id + 1)}")

        self.test_acc = test_acc / (batch_id + 1)

        # 결과물 출력
        result = pd.DataFrame(test_eval, columns=['topic_idx'])
        result['index'] = range(45654, 45654 + len(result))
        result.set_index('index', inplace=True)
        result.to_csv(f'{expid}_result_epoch{epoch}.csv')
        # display(result)

