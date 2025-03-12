
import os
import av
import urllib
import hydra
import pyrootutils
import soundfile as sf
from tqdm import tqdm
from typing import List, Tuple
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from argparse import ArgumentParser
from os.path import join, abspath
import yaml
from pathlib import Path
from glob import glob

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--noisy_dir", type=str, required=True, help='Path to the noisy directory')
    parser.add_argument("--video_roi_dir", type=str, required=True, help='Path to the video roi directory')
    parser.add_argument("--out_dir", type=str, default="out", help='Directory to save the model')
    parser.add_argument("--run_name", type=str, default="avgen", help='Directory containing the run logs')
    parser.add_argument("--ckpt", type=str, default="checkpoints/avgen.ckpt", help='Path to the checkpoint file')

    args = parser.parse_args()

    root_dir = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

    config_path = join("checkpoints", "hydra")
    
    cfg = yaml.safe_load(Path(join(config_path, "config.yaml")).read_text())
    hydra_cfg = yaml.safe_load(Path(join(config_path, "hydra.yaml")).read_text())
    hydra_cfg = OmegaConf.create(hydra_cfg["hydra"])

    # Download checkpoints if not present
    if not os.path.exists(args.ckpt):
        print("Downloading checkpoint...")
        CHECKPOINT_URL="https://www2.informatik.uni-hamburg.de/sp/audio/publications/avgen/avgen.ckpt"
        urllib.request.urlretrieve(CHECKPOINT_URL, "checkpoints/avgen.ckpt")

    print("Creating manifest file...")
    video_rois = glob(join(args.video_roi_dir, "**", "*.mp4"))
    noisy_wavs = glob(join(args.noisy_dir, "**", "*.wav"))

    target_manifest_path = join(args.out_dir, args.run_name, "manifest.tsv") 
    os.makedirs(os.path.dirname(target_manifest_path), exist_ok=True)

    with open(target_manifest_path, "w") as f:
        f.write("/\n")
        for noisy_wav in tqdm(noisy_wavs):
            file_name = noisy_wav.replace(args.noisy_dir, "").rstrip(".wav").lstrip("/")
            video_roi = join(args.video_roi_dir, file_name + ".mp4")
            audio, sr = sf.read(noisy_wav)
            assert sr == 16000, "Sample rate is not 16kHz"
            audio_len = len(audio)
            with av.open(video_roi) as container:
                video_stream = container.streams.video[0]
                num_frames = video_stream.frames
            f.write(f"{file_name}\t{video_roi}\t{noisy_wav}\t{num_frames}\t{audio_len}\n")

    print("Start inference...")

    cfg["ckpt_path"] = args.ckpt
    cfg["logger"] = None

    # Set min and max length of the audio files
    cfg["data"]["max_token_count"] = 700
    cfg["data"]["max_len"] = 700
    cfg["data"]["min_len"] = 0 # In the paper, we used min length of the audio files 3.2s (= 80 frames)
    cfg["data"]["_target_"] = "src.data.lrs_datamodule.LRS3DataModule"
    cfg["data"]["data_dir"] = "config"
    cfg["data"]["manifest_str"] = target_manifest_path
     
    cfg["model"]["params"]["log_dir"] = args.out_dir
    cfg["model"]["params"]["run_name"] = args.run_name

    # Resolve hydra config
    def hydra_resolver(x):
        elements = x.split(".")
        _hydra_cfg = hydra_cfg
        for element in elements:
            _hydra_cfg = _hydra_cfg[element]
        return _hydra_cfg
    OmegaConf.register_new_resolver("hydra", hydra_resolver)
    cfg = OmegaConf.create(cfg)

    evaluate(cfg)
    