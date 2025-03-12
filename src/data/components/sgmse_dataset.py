import torch
from torchaudio import load
import torch.nn.functional as F
import numpy as np
from glob import glob


def spec_fwd(spec, spec_abs_exponent=0.5, spec_factor=0.15):
    spec = spec.abs()**spec_abs_exponent * torch.exp(1j * spec.angle())
    spec = spec * spec_factor
    return spec

def get_sgmse_input(
        elem,
        n_fft=510,
        hop_length=160,
        normalize="not",
        center=True,
        window=None,
        spec_abs_exponent=0.5,
        spec_factor=0.15,
        **kwargs):
    
    stft_kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "center": center,
        "window": window,
        "return_complex": True,
    }

    clean_path = elem['clean_path']
    noisy_path = elem['noisy_path']

    x, _ = load(clean_path)
    y, _ = load(noisy_path)

    if normalize == "clean":
        x = x / x.abs().max()

    X = torch.stft(x, **stft_kwargs)
    Y = torch.stft(y, **stft_kwargs)

    X = spec_fwd(X, spec_abs_exponent, spec_factor)
    Y = spec_fwd(Y, spec_abs_exponent, spec_factor)

    return {"clean_spec": X, "noisy_spec": Y}


def get_sgmse_input_pred(
        elem,
        n_fft=510,
        hop_length=160,
        normalize="not",
        center=True,
        window=None,
        spec_abs_exponent=0.5,
        spec_factor=0.15,
        **kwargs):
    
    stft_kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "center": center,
        "window": window,
        "return_complex": True,
    }

    x, _ = load(elem['clean_path'])
    y, _ = load(elem['noisy_path']) 
    x_pred, _ = load(elem['pred_path'])

    if normalize == "clean":
        x = x / x.abs().max()

    X = torch.stft(x, **stft_kwargs)
    Y = torch.stft(y, **stft_kwargs)
    X_pred = torch.stft(x_pred, **stft_kwargs)

    X = spec_fwd(X, spec_abs_exponent, spec_factor)
    Y = spec_fwd(Y, spec_abs_exponent, spec_factor)
    X_pred = spec_fwd(X_pred, spec_abs_exponent, spec_factor)

    return {"clean_spec": X, "noisy_spec": Y, "pred_spec": X_pred}


def spec_collator(samples):
    spec_lens = [s.shape[-1] for s in samples]
    max_len = max(spec_lens)

    # pad to the nearest multiple of 64
    if max_len%64 != 0:
        target_len = max_len + 64 - max_len%64
    else:
        target_len = max_len

    pad_values  = target_len - np.array(spec_lens)

    padded_samples = []

    for i, s in enumerate(samples):
        if pad_values[i] > 0:
            pad2d = torch.nn.ZeroPad2d((0, pad_values[i], 0, 0))
            padded_samples.append(pad2d(s))
        else:
            padded_samples.append(s)

    padded_samples = torch.stack(padded_samples, dim=0)

    return padded_samples


def specs_collator(samples):
    clean_specs = [s['clean_spec'] for s in samples]
    noisy_specs = [s['noisy_spec'] for s in samples]
    
    clean_paths = [s['clean_path'] for s in samples]
    noisy_paths = [s['noisy_path'] for s in samples]

    clean_specs = spec_collator(clean_specs)
    noisy_specs = spec_collator(noisy_specs)

    return {
        "clean_specs": clean_specs, 
        "noisy_specs": noisy_specs, 
        "clean_paths": clean_paths,
        "noisy_paths": noisy_paths,
        }

def specs_collator_pred(samples):
    clean_specs = [s['clean_spec'] for s in samples]
    noisy_specs = [s['noisy_spec'] for s in samples]
    pred_specs = [s['pred_spec'] for s in samples]

    clean_paths = [s['clean_path'] for s in samples]
    noisy_paths = [s['noisy_path'] for s in samples]
    pred_paths = [s['pred_path'] for s in samples]

    clean_specs = spec_collator(clean_specs)
    noisy_specs = spec_collator(noisy_specs)
    pred_specs = spec_collator(pred_specs)

    return {
        "clean_specs": clean_specs, 
        "noisy_specs": noisy_specs, 
        "pred_specs": pred_specs,
        "clean_paths": clean_paths,
        "noisy_paths": noisy_paths,
        "pred_paths": pred_paths,
        }