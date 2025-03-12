import time
from math import ceil
import warnings

import torch
from lightning import LightningModule
from torch_ema import ExponentialMovingAverage
from lightning.pytorch.utilities import grad_norm

import numpy as np
import wandb
import matplotlib.pyplot as plt
import os
from os.path import join
from soundfile import write
import csv


# from sgmse import sampling
from src.models.sgmse.sdes import SDE
from src.models.sgmse.util.inference import evaluate_model
from src.models.sgmse.util.other import get_window
from src.models.sgmse.util.graphics import visualize_example
import src.models.sgmse.sampling as sampling

class AVGen(LightningModule):
    def __init__(
        self, 
        avhubert: torch.nn.Module,
        score_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        sde: SDE,
        params: dict,
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()

        self.avhubert = avhubert
        if params["freeze_avhubert"] == True:
            for param in self.avhubert.parameters():
                param.requires_grad = False
            self.avhubert.eval()

        self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sde = sde

        # Store hyperparams and save them
        self.ema_decay = params["ema_decay"]
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = params["t_eps"]
        self.num_eval_files = params["num_eval_files"]

        self.stft_kwargs = {
            "n_fft": params["n_fft"],
            "hop_length": params["hop_length"],
            "center": params["center"],
            "window": get_window(params["window_type"], params["n_fft"]),
        }

        self.params = params

        self.pesq = []; self.si_sdr = []; self.estoi = [] 

        self.save_hyperparameters(ignore=["no_wandb", "score_model", "avhubert"])

    def forward(self, x_t, t, cond):
        if "dropout_y" in self.params:
            if self.params["dropout_y"] > torch.rand(1).item() and self.training:
                y = torch.zeros_like(cond["noisy_specs"])
            else:
                y = cond["noisy_specs"]
        else: 
            y = cond["noisy_specs"]

        sigmas = self.sde._std(t)
        # Concatenate y as an extra channel
        if "without_y" in self.params:
            dnn_input = x_t
        else:
            dnn_input = torch.cat([x_t, y], dim=1)

        av_feats = self.avhubert.extract_finetune(source=cond['net_input']['source'], padding_mask=cond['net_input']['padding_mask'])
        
        score = self.score_model(dnn_input, t, sigmas, av_feats=av_feats['layer_results'])
        return score
    
    def on_train_start(self):
        pass

    def model_step(self, batch, batch_idx):
        x = batch['clean_specs']
        y = batch['noisy_specs']
        
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        x_t = mean + sigmas * z
        score = self(x_t, t, batch)
        err = score * sigmas + z  # implicit weighting by sigma squared (lambda in Song paper)
        loss = self._loss(err)
        return loss
    
    def _loss(self, err):
        losses = torch.square(err.abs())
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def on_before_optimizer_step(self, optimizer):
        pass
    
    def on_train_epoch_end(self):
        if "dropout_y" in self.params:
            if self.params["dropout_y"] > self.params["dropout_y_min"]:
                self.params["dropout_y"] = self.params["dropout_y"] - self.params["reduce_dropout_y_per_epoch"]


    def on_validation_epoch_start(self):
        self.trainer.val_dataloaders.seed(1)
        pass

    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self.pesq = []; self.si_sdr = []; self.estoi = [] 

        if batch_idx < self.num_eval_files:
            files, metrics = evaluate_model(self, batch)
            self.pesq.append(metrics[0]["pesq"])
            self.si_sdr.append(metrics[0]["si_sdr"])
            self.estoi.append(metrics[0]["estoi"])

            self._log_audio(files[0], batch_idx)
            self._log_spec(files[0], batch_idx)
        
        return loss
    
    def on_validation_epoch_end(self):
        self.log('pesq', np.array(self.pesq).mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('si_sdr', np.array(self.si_sdr).mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('estoi', np.array(self.estoi).mean(), on_step=False, on_epoch=True, sync_dist=True)

    def on_fit_end(self) -> None:
        pass


    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode=True, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)
    
    def on_test_start(self) -> None:
        self.trainer.test_dataloaders.seed(1)

        dir_name = join(self.params.log_dir, self.params.run_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


    def test_step(self, batch, batch_idx):
        files, metrics = evaluate_model(self, batch, first_only=False, with_metrics=False)

        for file, metric in zip(files, metrics):
            x_hat = file["x_hat"]
            noisy_path = file["noisy_path"]
            target_file_name = "/".join(noisy_path.split("/")[-4:])
            target_path = join(self.params.log_dir, self.params.run_name, target_file_name.replace("_target.wav", "_enhanced.wav"))

            dir_name = os.path.dirname(target_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            write(target_path, x_hat, self.params.sr)
            

        return metrics
    
    def on_test_end(self) -> None:
        pass

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}

        return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, **kwargs)

    def get_standard_sampler(self):
        sde = self.sde.copy()
        return sampling.get_standard_sampler(sde=sde, score_model=self)
       
    def to_audio(self, spec, length=None):
        return self.istft(self.spec_back(spec), length)

    def spec_fwd(self, spec):
        spec = spec.abs()**self.params["spec_abs_exponent"] * torch.exp(1j * spec.angle())
        spec = spec * self.params["spec_factor"]
        return spec
    
    def spec_back(self, spec):
        spec = spec / self.params["spec_factor"]
        spec = spec.abs()**(1/self.params["spec_abs_exponent"]) * torch.exp(1j * spec.angle())
        return spec

    def stft(self, sig):
        return torch.stft(sig, return_complex=True, **self.stft_kwargs)
    
    def istft(self, spec, length=None):
        self.stft_kwargs["window"] = self.stft_kwargs["window"].to(spec.device)
        return torch.istft(spec, length=length, **self.stft_kwargs)
    
