# :rabbit: [Detecting Voice Cloning Attacks via Timbre Watermarking](https://github.com/TimbreWatermarking/TimbreWatermarking)

Source code for [paper](https://www.ndss-symposium.org/wp-content/uploads/2024-200-paper.pdf) “Detecting Voice Cloning Attacks via Timbre Watermarking” 

by _Chang Liu, Jie Zhang, Tianwei Zhang, Xi Yang, Weiming Zhang, and Nenghai Yu_ 
In [Network and Distributed System Security Symposium (NDSS) 2024](https://www.ndss-symposium.org/ndss2024/).

Visit our [website](https://timbrewatermarking.github.io/samples.html) for audio samples.

## Introduction

:rabbit2: In this repository, we provide the complete code for training and testing the watermarking model. Additionally, we include the source code used for voice cloning experiments under various scenarios, along with corresponding README files. 

`Please visit the respective directories to access detailed READMEs.`

- [watermarking_model](https://github.com/TimbreWatermarking/TimbreWatermarking/tree/main/watermarking_model): Code of the watermarking model
- [voice.clone](https://github.com/TimbreWatermarking/TimbreWatermarking/tree/main/voice.clone): Code and details of the voice cloning part


## Install as a package & run inference

This fork packages the watermarking model so you can `pip install` it and embed /
decode watermarks from a few lines of Python — no need to run the training
scripts or wrangle config paths.

### Install

From the repository root:

```bash
# (recommended) create an environment, then install the package + its deps
pip install .

# PyTorch is left unpinned so you can match your CUDA/CPU setup — if you don't
# already have it, install the right build from https://pytorch.org first, e.g.
#   pip install torch torchaudio
```

> **numpy 1.x is required.** The compiled dependencies are built against numpy
> 1.x and crash under numpy 2.x, so the package pins `numpy<2`. If you installed
> packages by hand and ended up on numpy 2, run `pip install "numpy<2"`.

To also run training, dataset loading, or the speech-quality / speaker-similarity
metrics, install the extras:

```bash
pip install ".[full]"
```

The model checkpoints, HiFi-GAN vocoder, and YAML configs are bundled with the
package, so inference works out of the box.

### Inference

```python
import timbre

# Embed a watermark into an audio file (random watermark if none is given).
# Returns the watermarked waveform, its sample rate, and the embedded bits.
wav, sr, msg = timbre.add_watermark("input.wav", output_path="watermarked.wav")

# Decode the watermark back out (returns a 0/1 numpy array).
bits = timbre.decode_watermark("watermarked.wav")

print("embedded:", msg.tolist())
print("decoded :", bits.tolist())
print("bit accuracy:", timbre.bit_accuracy(msg, bits))
```

Embed a specific watermark (a list of bits, or a bit-string):

```python
timbre.add_watermark(
    "input.wav",
    watermark="101010101010101010101010101010",  # must match the model bit length
    output_path="watermarked.wav",
)
```

Reuse a loaded model to process many files efficiently, or pick a specific
checkpoint:

```python
tw = timbre.Timbre()                  # loads the default 30-bit checkpoint
print(tw.msg_bits, tw.device)         # e.g. 30  cuda

for path in my_audio_files:
    tw.add_watermark(path, output_path=path.replace(".wav", "_wm.wav"))

# Use a different checkpoint (file or directory; a directory picks the newest):
tw10 = timbre.Timbre(model_path="path/to/checkpoint.pth.tar")
```

`add_watermark` / `decode_watermark` also accept raw audio samples instead of a
file path — pass a numpy array or torch tensor together with its `sample_rate`
(audio is automatically resampled to the model's 22.05 kHz and run on GPU if
available).

> The default model is the bundled 30-bit checkpoint. The watermark **bit
> length is detected automatically from the checkpoint**, so the same code works
> with the 10-bit and 30-bit models — just make sure any explicit `watermark`
> you pass has the matching number of bits (`tw.msg_bits`).


## Model files
All the parameter files for the voice cloning model used in our work are available at [this link](https://drive.google.com/drive/folders/1tRbEneN1VsSCZ0HPxG3DSoJdxDRZ_NUJ?usp=drive_link).


## Acknowledgments

Part of our experiments were based on code from several open-source repositories, including [VITS](https://github.com/jaywalnut310/vits), [Tacotron2](https://github.com/NVIDIA/tacotron2), [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc), [Hifi-GAN](https://github.com/jik876/hifi-gan), and [FastSpeech2](https://github.com/ming024/FastSpeech2). Their code served as a foundation for portions of our experiments.



## Citation
If you find this work useful, please consider citing our paper:
```
@inproceedings{timbrewatermarking-ndss2024,
  title = {Detecting Voice Cloning Attacks via Timbre Watermarking},
  author = {Liu, Chang and Zhang, Jie and Zhang, Tianwei and Yang, Xi and Zhang, Weiming and Yu, Nenghai},
  booktitle = {Network and Distributed System Security Symposium},
  year = {2024},
  doi = {10.14722/ndss.2024.24200},
}
```
