import torch
from torchaudio import load

from pesq import pesq
from pystoi import stoi
from glob import glob

from .other import si_sdr, pad_spec

# Settings
sr = 16000
snr = 0.5
N = 30
corrector_steps = 1


def evaluate_model(model, batch, first_only=True, with_metrics=True):

    if first_only:
        # take first example in batch
        for key in batch:
            if key != "net_input":
                batch[key] = batch[key][0:1]
            else:
                if batch["net_input"]["source"]["audio"] is not None:
                    batch["net_input"]["source"]["audio"] = batch["net_input"]["source"]["audio"][0:1]
                batch["net_input"]["source"]["video"] = batch["net_input"]["source"]["video"][0:1]
                batch["net_input"]["padding_mask"] = batch["net_input"]["padding_mask"][0:1]

    clean_paths = batch['clean_paths']
    noisy_paths = batch['noisy_paths']

    sampler = model.get_standard_sampler()
    sample, _ = sampler(batch)

    num_samples = sample.size(0)

    files = []; metrics = []
    for i in range(num_samples):

        # Load wavs
        clean_path = clean_paths[i]
        noisy_path = noisy_paths[i]

        x, _ = load(clean_path)
        y, _ = load(noisy_path)
    
        T_orig = x.size(1)  

        x_hat = model.to_audio(sample[i, ...].squeeze(), T_orig)

        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        files.append({
            'x': x,
            'y': y,
            'x_hat': x_hat,
            'clean_path': clean_paths[i],
            'noisy_path': noisy_paths[i],
        })
        
        if with_metrics:
            metrics.append({
                'si_sdr': si_sdr(x, x_hat),
                'pesq': pesq(sr, x, x_hat, 'wb') ,
                'estoi': stoi(x, x_hat, sr, extended=True),
            })
        else:
            metrics.append({
            'si_sdr': 0.0,
            'pesq': 0.0,
            'estoi': 0.0,
            })
    return files, metrics
    
    
