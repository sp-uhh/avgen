import numpy as np
import torch
from .sgmse_dataset import spec_collator
from .hubert_dataset import collater_audio


def combined_collater(samples):
    # AV-Hubert feature rate is 25 Hz, SGMSE feature rate is 100 Hz
    downsampling_factor = 4
    
    if len(samples) == 0:
        return {}
    
    # collate input features for SGMSE
    clean_specs = [s['clean_spec'] for s in samples]
    noisy_specs = [s['noisy_spec'] for s in samples]

    clean_paths = [s['clean_path'] for s in samples]
    noisy_paths = [s['noisy_path'] for s in samples]

    collated_clean_specs = spec_collator(clean_specs)
    collated_noisy_specs = spec_collator(noisy_specs)

    # collate input features for AV-Hubert
    audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]

    spec_len = collated_clean_specs[0].shape[-1]

    # if audio_source[0] is not None:
    #     audio_sizes = [len(s) for s in audio_source]

    audio_size = int(spec_len / downsampling_factor)

    if audio_source[0] is not None:
        collated_audios, padding_mask, audio_starts = collater_audio(audio_source, audio_size)
    else:
        collated_audios, audio_starts = None, None
    if video_source is not None:
        collated_videos, padding_mask, audio_starts = collater_audio(video_source, audio_size, audio_starts)
    else:
        collated_videos = None

    
    source = {"audio": collated_audios, "video": collated_videos}
    net_input = {"source": source, "padding_mask": padding_mask}
    batch = {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "net_input": net_input,
        "utt_id": [s['fid'] for s in samples],
        "clean_specs": collated_clean_specs,
        "noisy_specs": collated_noisy_specs,
        "clean_paths": clean_paths,
        "noisy_paths": noisy_paths,
    }

    return batch


def combined_collater_pred(samples):
    # AV-Hubert feature rate is 25 Hz, SGMSE feature rate is 100 Hz
    downsampling_factor = 4
    
    if len(samples) == 0:
        return {}
    
    # collate input features for SGMSE
    clean_specs = [s['clean_spec'] for s in samples]
    noisy_specs = [s['noisy_spec'] for s in samples]
    pred_specs = [s['pred_spec'] for s in samples]

    clean_paths = [s['clean_path'] for s in samples]
    noisy_paths = [s['noisy_path'] for s in samples]
    pred_paths = [s['pred_path'] for s in samples]

    collated_clean_specs = spec_collator(clean_specs)
    collated_noisy_specs = spec_collator(noisy_specs)
    collated_pred_specs = spec_collator(pred_specs)

    # collate input features for AV-Hubert
    audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]

    spec_len = collated_clean_specs[0].shape[-1]

    # if audio_source[0] is not None:
    #     audio_sizes = [len(s) for s in audio_source]

    audio_size = int(spec_len / downsampling_factor)

    if audio_source[0] is not None:
        collated_audios, padding_mask, audio_starts = collater_audio(audio_source, audio_size)
    else:
        collated_audios, audio_starts = None, None
    if video_source is not None:
        collated_videos, padding_mask, audio_starts = collater_audio(video_source, audio_size, audio_starts)
    else:
        collated_videos = None

    
    source = {"audio": collated_audios, "video": collated_videos}
    net_input = {"source": source, "padding_mask": padding_mask}
    batch = {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "net_input": net_input,
        "utt_id": [s['fid'] for s in samples],
        "clean_specs": collated_clean_specs,
        "noisy_specs": collated_noisy_specs,
        "pred_specs": collated_pred_specs,
        "clean_paths": clean_paths,
        "noisy_paths": noisy_paths,
        "pred_paths": pred_paths,
    }

    return batch