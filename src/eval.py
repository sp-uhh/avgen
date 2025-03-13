import os
import av
import yaml
import urllib
import hydra
import pyrootutils
import soundfile as sf

from glob import glob
from tqdm import tqdm
from os.path import join
from pathlib import Path
from omegaconf import OmegaConf
from argparse import ArgumentParser
from lightning import LightningDataModule, LightningModule, Trainer

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def evaluate(cfg):
    assert cfg.ckpt_path

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=False)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    print("Testing finished!")
    print("Teardown...")
    datamodule.teardown()


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

    video_rois = sorted(glob(join(args.video_roi_dir, "**", "*.mp4")))
    video_rois += sorted(glob(join(args.video_roi_dir, "*.mp4")))
    noisy_wavs = sorted(glob(join(args.noisy_dir, "**", "*.wav")))
    noisy_wavs += sorted(glob(join(args.noisy_dir, "*.wav")))

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

    # Set min and max length of the audio files
    cfg["data"]["max_token_count"] = 700
    cfg["data"]["max_len"] = 700
    cfg["data"]["min_len"] = 0 # In the paper, we used min length of the audio files 3.2s (= 80 frames)
    cfg["data"]["_target_"] = "src.data.lrs_datamodule.LRS3DataModule"
    cfg["data"]["data_dir"] = "config"
    cfg["data"]["manifest_str"] = target_manifest_path
    cfg["model"]["params"]["log_dir"] = args.out_dir
    cfg["model"]["params"]["run_name"] = args.run_name
    cfg["model"]["params"]["noisy_dir"] = args.noisy_dir
    cfg["ckpt_path"] = args.ckpt

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
    