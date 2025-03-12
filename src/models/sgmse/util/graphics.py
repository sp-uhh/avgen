import torch
from torchaudio import load
import matplotlib.pyplot as plt
import numpy as np
import os

# Plotting settings
EPS_graphics = 1e-10
n_fft = 512
hop_length = 128
vmin, vmax = -60, 0

stft_kwargs = {"n_fft": n_fft, "hop_length": hop_length, "window": torch.hann_window(n_fft), "center": True, "return_complex": True}

def visualize_example(x):
	"""Visualize training targets and estimates of the Neural Network
	Args:
		- mix: Tensor [F, T]
		- estimates/targets: Tensor [F, T]
	"""

	hop_length = 256
	n_fft = 1024
	sr = 16000

	spec = torch.stft(torch.tensor(x), n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(1024), return_complex=True)
	spec = spec.numpy()
	spec = np.abs(spec)
	spec = 20*np.log10(spec + 1e-8)

	T_coef = np.arange(spec.shape[1]) * hop_length / sr
	F_coef = np.arange(spec.shape[0]) * sr / n_fft

	left = min(T_coef)
	right = max(T_coef) + n_fft / sr
	lower = min(F_coef)/1000
	upper = max(F_coef)/1000

	fig, ax = plt.subplots(1, 1, figsize=(6, 4))

	img1 = ax.imshow(spec, origin="lower", aspect='auto', cmap='magma', vmin=-40, vmax=20, extent=(left, right, lower, upper))
	ax.set_ylabel("Frequency [kHz]")
	ax.set_xlabel("Time [s]")
	fig.colorbar(img1, ax=ax)

	return fig