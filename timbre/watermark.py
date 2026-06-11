"""High-level API for the Timbre watermarking model.

This module wraps the encoder/decoder defined in
``timbre.model.conv2_mel_modules`` behind a small, friendly interface so that
callers can do::

    import timbre

    tw = timbre.Timbre()                      # loads the latest checkpoint
    out, sr, msg = tw.add_watermark("in.wav", output_path="out.wav")
    bits = tw.decode_watermark("out.wav")
    print(timbre.bit_accuracy(msg, bits))

Or, without managing an object::

    out, sr, msg = timbre.add_watermark("in.wav", output_path="out.wav")
    bits = timbre.decode_watermark("out.wav")

The watermark length (number of bits) is detected automatically from the
checkpoint, so the same code works with the 10-bit and 30-bit models.

Nothing here changes the behaviour of the existing scripts
(``embed_and_save.py``, ``extract.py``, training, ...) -- it only adds a new
entry point that reuses the same model code.
"""

import os
import contextlib

import numpy as np
import torch
import yaml

# Directory that contains this package (so the API works regardless of the
# caller's current working directory).
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.join(_PKG_DIR, "config")
_DEFAULT_CKPT_DIR = os.path.join(_PKG_DIR, "results", "ckpt", "pth")

# Checkpoint used when none is given. This is the best-performing committed model
# (30-bit, ~0.91 round-trip bit accuracy). Other checkpoints in the repo are an
# undertrained work-in-progress (newest by date) and the original 10-bit model.
# Pass ``model_path=`` to load a different file or directory.
_DEFAULT_CKPT_NAME = "compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar"

# Target sample rate the model was trained at (process.yaml -> audio.sample_rate).
_MODEL_SAMPLE_RATE = 22050


@contextlib.contextmanager
def _chdir(path):
    """Temporarily switch the working directory.

    The vocoder loader in ``conv2_mel_modules.get_vocoder`` reads a couple of
    paths relative to the current working directory (``hifigan/config.json`` and
    ``hifigan/model/VCTK_V1/generator_v1``). We run model construction from
    inside the package directory so it works no matter where the caller is.
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _resolve_checkpoint(model_path):
    """Return an absolute path to a checkpoint file.

    ``model_path`` may be a direct ``.pth.tar`` file or a directory. When it is
    ``None`` the default checkpoint (:data:`_DEFAULT_CKPT_NAME`) is used. When it
    is a directory, the most recently modified checkpoint inside it is used --
    the same selection the existing scripts make with ``index=-1``.
    """
    if model_path is None:
        default = os.path.join(_DEFAULT_CKPT_DIR, _DEFAULT_CKPT_NAME)
        if os.path.isfile(default):
            return default
        # Fall back to scanning the default directory if the preferred file is
        # missing (e.g. a custom checkpoint layout).
        model_path = _DEFAULT_CKPT_DIR
    if os.path.isfile(model_path):
        return model_path
    if os.path.isdir(model_path):
        ckpts = [f for f in os.listdir(model_path) if f.endswith((".pth.tar", ".pth"))]
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint (*.pth.tar) found in {model_path!r}")
        ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
        return os.path.join(model_path, ckpts[-1])
    raise FileNotFoundError(f"Checkpoint path does not exist: {model_path!r}")


def _detect_msg_bits(state, fallback):
    """Infer the watermark length from a decoder state dict."""
    key = "msg_linear_out.fc_layer.fc_layer.linear.weight"
    decoder_state = state.get("decoder", state)
    if key in decoder_state:
        return int(decoder_state[key].shape[0])
    return fallback


def _to_waveform_tensor(audio, sample_rate, device):
    """Normalise an audio input to a ``(1, 1, T)`` float tensor at the model rate.

    ``audio`` may be a path to a wav file, a 1-D/2-D numpy array, or a torch
    tensor. When an array/tensor is given, ``sample_rate`` must be provided.
    """
    import torchaudio

    if isinstance(audio, str):
        wav, sr = torchaudio.load(audio)
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing raw audio samples")
        if isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio).float()
        elif torch.is_tensor(audio):
            wav = audio.float()
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)!r}")
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3:
            wav = wav.squeeze(0)
        sr = sample_rate

    # Mono: keep a single channel as (1, T).
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != _MODEL_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, _MODEL_SAMPLE_RATE)

    # (1, T) -> (1, 1, T)
    return wav.unsqueeze(0).to(device)


class Timbre:
    """Loads the watermarking model once and embeds / extracts watermarks."""

    def __init__(self, model_path=None, device=None, msg_bits=None,
                 process_config=None, model_config=None):
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        process_config = process_config or _load_yaml(os.path.join(_CONFIG_DIR, "process.yaml"))
        model_config = model_config or _load_yaml(os.path.join(_CONFIG_DIR, "model.yaml"))
        self.sample_rate = process_config["audio"]["sample_rate"]

        ckpt_path = _resolve_checkpoint(model_path)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model_path = ckpt_path
        self.msg_bits = msg_bits or _detect_msg_bits(state, fallback=10)

        # Build the encoder/decoder. We import the model lazily (and build it
        # from inside the package dir) because the decoder constructs the
        # HiFi-GAN vocoder, which reads config files via relative paths.
        from timbre.model.conv2_mel_modules import Encoder, Decoder

        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_encoder = model_config["layer"]["nlayers_encoder"]
        nlayers_decoder = model_config["layer"]["nlayers_decoder"]
        heads_encoder = model_config["layer"]["attention_heads_encoder"]
        heads_decoder = model_config["layer"]["attention_heads_decoder"]

        with _chdir(_PKG_DIR):
            encoder = Encoder(
                process_config, model_config, self.msg_bits, win_dim, embedding_dim,
                nlayers_encoder=nlayers_encoder, attention_heads=heads_encoder,
            ).to(self.device)
            decoder = Decoder(
                process_config, model_config, self.msg_bits, win_dim, embedding_dim,
                nlayers_decoder=nlayers_decoder, attention_heads=heads_decoder,
            ).to(self.device)

        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"], strict=False)
        encoder.eval()
        decoder.eval()
        decoder.robust = False

        self.encoder = encoder
        self.decoder = decoder

    # -- watermark helpers --------------------------------------------------

    def random_watermark(self):
        """Return a random watermark of the right length as a 0/1 numpy array."""
        return np.random.randint(0, 2, size=self.msg_bits).astype(np.int64)

    def _prepare_msg(self, watermark):
        """Validate a 0/1 watermark and map it to the model's {-1, +1} tensor."""
        if watermark is None:
            watermark = self.random_watermark()
        if isinstance(watermark, str):
            watermark = [int(c) for c in watermark if c in "01"]
        bits = np.asarray(watermark, dtype=np.int64).reshape(-1)
        if bits.size != self.msg_bits:
            raise ValueError(
                f"watermark must have {self.msg_bits} bits, got {bits.size}"
            )
        if not np.isin(bits, (0, 1)).all():
            raise ValueError("watermark must contain only 0s and 1s")
        msg = torch.from_numpy(bits).float().reshape(1, 1, -1) * 2 - 1
        return bits, msg.to(self.device)

    # -- public API ---------------------------------------------------------

    @torch.no_grad()
    def add_watermark(self, audio, watermark=None, sample_rate=None, output_path=None):
        """Embed ``watermark`` into ``audio``.

        Parameters
        ----------
        audio : str | np.ndarray | torch.Tensor
            Path to a wav file, or raw mono samples (with ``sample_rate`` set).
        watermark : sequence of {0,1} | str | None
            The bits to embed. ``None`` generates a random watermark.
        sample_rate : int | None
            Sample rate of ``audio`` when raw samples are given.
        output_path : str | None
            If set, the watermarked audio is written here as a wav file.

        Returns
        -------
        (waveform, sample_rate, watermark) : (np.ndarray, int, np.ndarray)
            The watermarked mono waveform (float32), its sample rate, and the
            0/1 watermark that was embedded.
        """
        bits, msg = self._prepare_msg(watermark)
        x = _to_waveform_tensor(audio, sample_rate, self.device)

        encoded, _ = self.encoder.test_forward(x, msg)
        wav = encoded.squeeze(0).squeeze(0).detach().cpu().numpy()

        if output_path is not None:
            import soundfile
            soundfile.write(output_path, wav, samplerate=self.sample_rate)

        return wav, self.sample_rate, bits

    @torch.no_grad()
    def decode_watermark(self, audio, sample_rate=None, return_logits=False):
        """Extract the watermark from ``audio``.

        Parameters
        ----------
        audio : str | np.ndarray | torch.Tensor
            Path to a wav file, or raw mono samples (with ``sample_rate`` set).
        sample_rate : int | None
            Sample rate of ``audio`` when raw samples are given.
        return_logits : bool
            If True, also return the raw decoder output (before thresholding).

        Returns
        -------
        np.ndarray
            The decoded watermark as a 0/1 array of length ``msg_bits``. When
            ``return_logits`` is True, returns ``(bits, logits)``.
        """
        x = _to_waveform_tensor(audio, sample_rate, self.device)
        decoded = self.decoder.test_forward(x)
        logits = decoded.squeeze().detach().cpu().numpy()
        bits = (logits >= 0).astype(np.int64)
        if return_logits:
            return bits, logits
        return bits


def bit_accuracy(reference, decoded):
    """Fraction of matching bits between two 0/1 watermarks (in ``[0, 1]``)."""
    reference = np.asarray(reference).reshape(-1)
    decoded = np.asarray(decoded).reshape(-1)
    if reference.size != decoded.size:
        raise ValueError("watermarks must have the same length to compare")
    return float((reference == decoded).mean())


# -- module-level convenience wrappers (lazy singleton) ---------------------

_DEFAULT_MODEL = None


def load_model(model_path=None, device=None, **kwargs):
    """Build (or rebuild) and return the shared default :class:`Timbre` model."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = Timbre(model_path=model_path, device=device, **kwargs)
    return _DEFAULT_MODEL


def _get_default_model():
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = Timbre()
    return _DEFAULT_MODEL


def add_watermark(audio, watermark=None, sample_rate=None, output_path=None):
    """Embed a watermark using the shared default model. See :meth:`Timbre.add_watermark`."""
    return _get_default_model().add_watermark(
        audio, watermark=watermark, sample_rate=sample_rate, output_path=output_path
    )


def decode_watermark(audio, sample_rate=None, return_logits=False):
    """Extract a watermark using the shared default model. See :meth:`Timbre.decode_watermark`."""
    return _get_default_model().decode_watermark(
        audio, sample_rate=sample_rate, return_logits=return_logits
    )
