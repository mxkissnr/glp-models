# glp-models

Model weights for the on-device bean photo cut-out ("sticker") feature of
[Gaggiuino Local Profiler](https://github.com/mxkissnr/gaggiuino-local-profiler).
The app's Docker build downloads these files from this repository's releases,
verifies their SHA-256 and serves them locally. Nothing is fetched from a
third party at runtime and the cut-out runs entirely in the browser.

The files are attached to GitHub releases, not committed to git.

## Release `models-v1`

| File | Bytes | SHA-256 | Source |
|---|---|---|---|
| `isnet-general-use-int8.onnx` | 46360717 | `f1b1c6f7656e532627697afc989d953be1e7ef8f55a718f3611e8c9fd50cdef7` | IS-Net general-use, quantized (below) |
| `slimsam-vision-encoder-q8.onnx` | 8882165 | `cce23c7b2e5d4f330932738fb67ba518e04b0d99ccdd1cccd22a7da4e01f2971` | `Xenova/slimsam-77-uniform` `onnx/vision_encoder_quantized.onnx` |
| `slimsam-decoder-q8.onnx` | 4903810 | `cb90b279f549d2cab7fd6e20c38522438c65d84bdcca3d2a764cff7d857fdce2` | `Xenova/slimsam-77-uniform` `onnx/prompt_encoder_mask_decoder_quantized.onnx` |

SlimSAM files are byte-identical to Hugging Face revision
`5850ab45f587c112167512ffef949107115e26a0`.

### How the IS-Net file was made

Input: `isnet-general-use.onnx` from rembg release `v0.0.0`
(178648008 bytes, SHA-256 `60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a`).

Quantized with onnxruntime 1.30.0:

```
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("isnet-general-use.onnx", "isnet-general-use-int8.onnx", weight_type=QuantType.QUInt8)
```

## Licence

All weights are under the Apache License 2.0, see `LICENSE` and `NOTICE`.

## Security

Every release asset is checked in CI on each push, each release and weekly: checksum, the official ONNX checker, an operator allowlist and a load test in onnxruntime. Details and how to report a problem: `SECURITY.md`.
