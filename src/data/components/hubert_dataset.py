from python_speech_features import logfbank
import numpy as np
from scipy.io import wavfile
import cv2
import random
import torch
import torch.nn.functional as F


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)

class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


def _load_video(path):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")


image_crop_size = 88
image_mean = 0.421
image_std = 0.165

transform = Compose([
                Normalize(0.0, 255.0),
                CenterCrop((image_crop_size, image_crop_size)),
                Normalize(image_mean, image_std) 
                ])


def load_video(video_path):
        feats = _load_video(video_path)
        feats = transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

def get_avhubert_input(elem, modalities=['audio', 'video'], stack_order_audio=4, noise_prob=0.0, normalize=True):
        """
        Taken and modified from AV-HUBERT repo

        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        
        fid = elem['fid']
        video_path = elem['video_path']
        audio_path = elem['noisy_path']
        if 'video' in modalities:
            video_feats = load_video(video_path) # [T, H, W, 1]
        else:
            video_feats = None
        if 'audio' in modalities:
            sample_rate, wav_data = wavfile.read(audio_path)
            assert sample_rate == 16_000 and len(wav_data.shape) == 1
            # if np.random.rand() < noise_prob:
            #     wav_data = self.add_noise(wav_data)
            audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
            audio_feats = stacker(audio_feats, stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
        else:
            audio_feats = None

        # cut audio and video to the same length
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            # if video is longer, pad audio
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            # if audio is longer, cut audio
            elif diff > 0:
                audio_feats = audio_feats[:-diff]

        audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        if normalize and 'audio' in modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        # labels = get_labels(index) #TODO
        index = 0 # TODO
        labels = None # TODO
        return {"id": index, 'fid': fid, "video_source": video_feats, 'audio_source': audio_feats, "label_list": labels}


def avhubert_collater(samples, pad_audio=True, max_sample_size=500):
    samples = [s for s in samples if s["id"] is not None]
    if len(samples) == 0:
        return {}

    audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]
    if audio_source[0] is None:
        audio_source = None
    if video_source[0] is None:
        video_source = None
    if audio_source is not None:
        audio_sizes = [len(s) for s in audio_source]
    else:
        audio_sizes = [len(s) for s in video_source]
    if pad_audio:
        audio_size = min(max(audio_sizes), max_sample_size)
    else:
        audio_size = min(min(audio_sizes), max_sample_size)
    if audio_source is not None:
        collated_audios, padding_mask, audio_starts = collater_audio(audio_source, audio_size)
    else:
        collated_audios, audio_starts = None, None
    if video_source is not None:
        collated_videos, padding_mask, audio_starts = collater_audio(video_source, audio_size, audio_starts)
    else:
        collated_videos = None
    # targets_by_label = [
    #     [s["label_list"][i] for s in samples]
    #     for i in range(self.num_labels)
    # ]
    # targets_list, lengths_list, ntokens_list = self.collater_label(
    #     targets_by_label, audio_size, audio_starts
    # )
    source = {"audio": collated_audios, "video": collated_videos}
    net_input = {"source": source, "padding_mask": padding_mask}
    batch = {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "net_input": net_input,
        "utt_id": [s['fid'] for s in samples],
    }

    # if self.single_target:
    #     batch["target_lengths"] = lengths_list[0]
    #     batch["ntokens"] = ntokens_list[0]
    #     if self.is_s2s:
    #         batch['target'], net_input['prev_output_tokens'] = targets_list[0][0], targets_list[0][1]
    #     else:
    #         batch["target"] = targets_list[0]
    # else:
    #     batch["target_lengths_list"] = lengths_list
    #     batch["ntokens_list"] = ntokens_list
    #     batch["target_list"] = targets_list
    return batch

def collater_audio(audios, audio_size, audio_starts=None, pad_audio=True):
    audio_feat_shape = list(audios[0].shape[1:])
    collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
    padding_mask = (
        torch.BoolTensor(len(audios), audio_size).fill_(False) # 
    )
    start_known = audio_starts is not None
    audio_starts = [0 for _ in audios] if not start_known else audio_starts
    for i, audio in enumerate(audios):
        diff = len(audio) - audio_size
        if diff == 0:
            collated_audios[i] = audio
        elif diff < 0:
            assert pad_audio
            collated_audios[i] = torch.cat(
                [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
            )
            padding_mask[i, diff:] = True
        else:
            collated_audios[i], audio_starts[i] = crop_to_max_size(
                audio, audio_size, audio_starts[i] if start_known else None
            )
    if len(audios[0].shape) == 2:
        collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
    else:
        collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
    return collated_audios, padding_mask, audio_starts


def crop_to_max_size(wav, target_size, start=None, random_crop=False):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start