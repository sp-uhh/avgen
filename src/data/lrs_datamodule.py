from os.path import join
from typing import Any, Dict, Optional, Callable
from functools import partial

import torch
from lightning import LightningDataModule
from torchdata.datapipes.iter import IterDataPipe, FileOpener, IterableWrapper
from torchdata.dataloader2 import (DataLoader2, MultiProcessingReadingService, 
                                   DistributedReadingService, SequentialReadingService) 


import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components import get_avhubert_input, get_sgmse_input, combined_collater, specs_collator


class LRS3DataPipe(IterDataPipe):
    def __init__(self, 
                 data_dir: str,
                 manifest_path: str,
                 max_token_count: int, # corresponds max video frames to collate into a batch
                 min_len: int,
                 max_len: int,
                 load_items_fn: Callable,
                 collate_fn: Callable,
                 shuffle: bool = True,
                 ):
        
        self.dp = FileOpener([manifest_path])
        self.dp = self.dp.parse_csv(skip_lines=1, delimiter="\t")
        # map csv rows to dict
        self.dp = self.dp.map(map_to_dict)
        # bucketize over the whole dataset
        self.dp = self.dp.max_token_bucketize2(max_token_count=max_token_count, 
                                     min_len=min_len, 
                                     max_len=max_len, 
                                     len_fn=get_video_len, 
                                     buffer_size=len(list(self.dp)),
                                     include_padding=True,)    
        
        if shuffle:
            self.dp = self.dp.shuffle(buffer_size=len(list(self.dp)))
        self.dp = self.dp.sharding_filter()
        # load items
        self.dp = self.dp.map(load_items_fn)
        # collate items
        self.dp = self.dp.collate(collate_fn=collate_fn)

    def __iter__(self):
        return iter(self.dp)
    
    def __len__(self):
        return len(list(self.dp))


def load_items(
        x,
        sgmse_kwargs: Dict[str, Any] = {}, 
        avhubert_kwargs: Dict[str, Any] = {},
        audio_only: bool = False,
    ):
    combined_feats = []
    # if audio_only load only features for SGMSE
    if audio_only:
        for elem in x:
            combined_feats.append(
                dict(
                    **get_sgmse_input(elem, **sgmse_kwargs),
                    clean_path=elem["clean_path"],
                    noisy_path=elem["noisy_path"], 
                )
            )
        return combined_feats
    else:
        for elem in x:
            combined_feats.append(
                dict(
                    **get_sgmse_input(elem, **sgmse_kwargs),
                    **get_avhubert_input(elem, **avhubert_kwargs),
                    clean_path=elem["clean_path"],
                    noisy_path=elem["noisy_path"], 
                )
            )
    return combined_feats

def get_video_len(x):
    return x["video_length"]

def map_to_dict(x):
    return {
        "fid": x[0], 
        "video_path": x[1],
        "clean_path": x[2],
        "noisy_path": x[2],
        "video_length": int(x[3]), 
        "audio_length": int(x[4])
        }

def map_to_video_len(x):
    return {"video_length": int(x[3])}


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class LRS3DataModule(LightningDataModule):
    """DataModule for LRS3 dataset.

    Args:
        data_dir (str, optional): Path to data directory. Defaults to "data/".
    """  
    def __init__(
        self,
        data_dir: str,
        manifest_str: str,
        max_token_count: int = 300,
        min_len: int = 80,
        max_len: int = 300,
        num_workers: int = 4,
        audio_only: bool = False,
        sgmse_kwargs = {},
        avhubert_kwargs = {},
    ):
        super().__init__()

        self.data_dir = data_dir
        self.manifest_str = manifest_str
        self.num_workers = num_workers
        self.min_len = min_len
        self.max_len = max_len
        self.max_token_count = max_token_count

        self.sgmse_kwargs = sgmse_kwargs
        self.sgmse_kwargs.update({"window": get_window(sgmse_kwargs["window_type"], sgmse_kwargs["n_fft"])})
        self.avhubert_kwargs = avhubert_kwargs
        self.distributed = torch.cuda.device_count() > 1

        self.load_items = partial(load_items, 
                                  sgmse_kwargs=self.sgmse_kwargs, 
                                  avhubert_kwargs=self.avhubert_kwargs,
                                  audio_only=audio_only,
                                  )
        
        if audio_only:
            self.collate_fn = specs_collator
        else:  
            self.collate_fn = combined_collater


        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.train_dp`, `self.val_dp`, `self.test_dp`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if stage == 'fit' or stage is None:
            self.train_dp = LRS3DataPipe(
                data_dir=self.data_dir,
                manifest_path=join(self.data_dir, self.manifest_str, "train.tsv"),
                max_token_count=self.max_token_count,
                min_len=self.min_len,
                max_len=self.max_len,
                load_items_fn=self.load_items,
                collate_fn=self.collate_fn,
                )
            self.val_dp = LRS3DataPipe(
                data_dir=self.data_dir,
                manifest_path=join(self.data_dir, self.manifest_str, "valid.tsv"),
                max_token_count=self.max_token_count,
                min_len=self.min_len,
                max_len=self.max_len,
                load_items_fn=self.load_items,
                collate_fn=self.collate_fn,
                shuffle=True,
                )
        if stage == 'test' or stage is None:
            self.test_dp = LRS3DataPipe(
                    data_dir=self.data_dir,
                    manifest_path=self.manifest_str,
                    max_token_count=self.max_token_count,
                    min_len=self.min_len,
                    max_len=self.max_len,
                    load_items_fn=self.load_items,
                    collate_fn=self.collate_fn,
                    )

    def train_dataloader(self):
        # return DataLoader2(self.train_dp)
        rs = MultiProcessingReadingService(self.num_workers, multiprocessing_context='spawn')
        if self.distributed:
            rs = SequentialReadingService(DistributedReadingService(), rs)
        return DataLoader2(self.train_dp, reading_service=rs)


    def val_dataloader(self):
        rs = MultiProcessingReadingService(self.num_workers, multiprocessing_context='spawn')
        if self.distributed:
            rs = SequentialReadingService(DistributedReadingService(), rs)
        dl = DataLoader2(self.val_dp, reading_service=rs)
        dl.seed(1)
        return dl

    def test_dataloader(self):
        rs = MultiProcessingReadingService(self.num_workers, multiprocessing_context='spawn')
        if self.distributed:
            rs = SequentialReadingService(DistributedReadingService(), rs)
        dl = DataLoader2(self.test_dp, reading_service=rs)
        dl.seed(1)
        return dl


    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


    @property
    def stft_kwargs(self):
        return {
            "n_fft": self.sgmse_kwargs["n_fft"],
            "hop_length": self.sgmse_kwargs["hop_length"],
            "window": self.sgmse_kwargs["window"],
            "center": self.sgmse_kwargs["center"], 
            "return_complex": True
            }

    @property
    def istft_kwargs(self):
        return {
            "n_fft": self.sgmse_kwargs["n_fft"],
            "hop_length": self.sgmse_kwargs["hop_length"],
            "window": self.sgmse_kwargs["window"],
            "center": self.sgmse_kwargs["center"],
        }


if __name__ == "__main__":
    pass
