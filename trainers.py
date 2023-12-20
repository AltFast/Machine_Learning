import torch
import os
import json
import random
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from configs import TrainingConfig
# New import by ME
import numpy as np
import pandas as pd
from utils import logprobs_from_logits
from early_stopping import EarlyStopping

class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        with open(f'./runs/{self.run_name}/metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.model.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')

    def save_loss(self, Loss_Train, Loss_Val):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        # Convert the list of NumPy arrays to a pandas DataFrame
        df = pd.DataFrame({'Loss_Train': Loss_Train, 'Loss_Val': Loss_Val})

        # Save the DataFrame to a text file (change 'output.txt' to your desired file name)
        df.to_csv(f'./runs/{self.run_name}/loss.txt', sep='\t', index=False, header=False)

class SFTTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.run_name = f"sft_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"

        # create a dataloader
        # get a batch of (data, label) by: x, y = self.train_dataloader
        self.train_dataloader = iter(
            DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.test_dataloader = iter(
            DataLoader(test_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.model = model
        self.dtype = torch.float16

        self.finetune_method = cfg.finetune_method

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        # TODO: complete the SFT training.
        stopper = EarlyStopping(self)
        eval_samples = 200    # 200 samples for each evaluation
        eval_interval = 500  # evaluate model every 500 iteration
        Loss_Train = []
        Loss_Val = []

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.model = self.model.to(self.device)
        
        for iter in range(self.cfg.max_steps):
            
            if iter % eval_interval == 0:  # Evaluate train error and test error
                Er_in, Er_out = self.estimate_loss(eval_samples)
                print(f"step {iter}: train loss {Er_in:.4f}, val loss {Er_out:.4f}")
                Loss_Train.append(Er_in.cpu().item())
                Loss_Val.append(Er_out.cpu().item())
                stopper(Er_out)
                if stopper.early_stop:
                    self.save_loss(Loss_Train, Loss_Val)
                    print("Early Stopping...")
                    return Loss_Train, Loss_Val
                    # break
                elif iter == self.cfg.max_steps - 1:
                    self.save_states(self.cfg.max_steps, True)
                    self.save_loss(Loss_Train, Loss_Val)
                    print("Normal Stopping...Model saved.")
                    return Loss_Train, Loss_Val
                    
            x, y = next(self.train_dataloader)
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = - logprobs_from_logits(logits, y).mean()

            if iter % 10 == 0: print(f"step {iter}: train loss {loss.item():.4f}")
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # self.save_states(self.cfg.max_steps, is_last=True)
        # self.save_loss(Loss_Train, Loss_Val)
        return Loss_Train, Loss_Val

    def estimate_loss(self, eval_samples):
        with torch.no_grad():

            self.model = self.model.to(self.device)
            self.model.eval()

            Er_in = torch.zeros(eval_samples)
            for k in range(eval_samples):
                x, y = next(self.train_dataloader)
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = - logprobs_from_logits(logits, y).mean()
                Er_in[k] = loss.item()

            Er_out = torch.zeros(eval_samples)
            for k in range(eval_samples):
                x, y = next(self.test_dataloader)
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = - logprobs_from_logits(logits, y).mean()
                Er_out[k] = loss.item()

            self.model.train()

        return Er_in.mean(), Er_out.mean()






