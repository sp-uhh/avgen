# Audio-Visual Speech Enhancement with Score-Based Generative Models

This repository contains the inference code for the paper ["Audio-Visual Speech Enhancement with Score-Based Generative Models"](https://ieeexplore.ieee.org/document/10363042/). 

## Setup

A `requirements.txt` file has been created using `pip freeze` from the virtual environment. Not all packages listed may be necessary for your use. To install all the packages, use the following command:

```bash
pip install -r requirements.txt
```

## Inference

To run the inference code, use the following command:

```bash
python src/eval.py --noisy_dir $noisy_fir --video_roi_dir $video_roi_dir --out_dir $out_dir
```

where `$noisy_dir` is the directory containing the noisy audio files and `$video_roi_dir` is the directory containing the video ROI files. The output will be saved in the directory `$out_dir`.

Make sure that the noisy audio and video files are named the same way, e.g., `audio_0001.wav` and `video_0001.mp4`.

## Citation

If you find this code useful, please consider citing the following paper:

```bib
@inproceedings{richter2023audio,
  title={Audio-visual speech enhancement with score-based generative models},
  author={Richter, Julius and Frintrop, Simone and Gerkmann, Timo},
  booktitle={Proceedings of ITG Conference on Speech Communication},
  pages={275--279},
  doi={10.30420/456164054},
  year={2023}
}
```