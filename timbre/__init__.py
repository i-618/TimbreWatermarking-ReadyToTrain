"""Timbre watermarking package.

The high-level watermarking API lives in :mod:`timbre.watermark` and is exposed
here lazily so that ``import timbre`` stays cheap and existing imports such as
``from timbre.model.loss import Loss`` keep working without pulling in torch up
front. The heavy modules are only imported the first time you touch the API::

    import timbre

    out, sr, msg = timbre.add_watermark("in.wav", output_path="out.wav")
    bits = timbre.decode_watermark("out.wav")
    print("bit accuracy:", timbre.bit_accuracy(msg, bits))

    # or keep a model around to embed/decode many files efficiently
    tw = timbre.Timbre()
    tw.add_watermark("in.wav", watermark=[1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                     output_path="out.wav")
"""

_LAZY = {"Timbre", "add_watermark", "decode_watermark", "load_model", "bit_accuracy"}

__all__ = sorted(_LAZY)


def __getattr__(name):
    # PEP 562: defer importing the watermarking implementation (and torch)
    # until one of the public API names is actually accessed.
    if name in _LAZY:
        from . import watermark
        return getattr(watermark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY)
