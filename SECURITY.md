# Security Policy

## Supported Versions

Only the **latest release** (`models-vN`) is supported. Gaggiuino Local Profiler pins the exact release and SHA-256 of every file it downloads.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please open a **[private security advisory](https://github.com/mxkissnr/glp-models/security/advisories/new)** and include a description, steps to reproduce and the potential impact. I will acknowledge your report within **7 days**.

## What is checked

Every release asset is validated in CI (`scripts/validate.py`) on each push, each release and weekly:

- SHA-256 matches `SHA256SUMS`, and no unlisted model file is present
- the file passes the official ONNX checker (`full_check`)
- only standard ONNX operators plus a short allowlist of onnxruntime contrib ops; no custom operator domains, local functions or training info
- no tensor references external data on disk
- onnxruntime can load the model

ONNX files are protobuf graph descriptions, not executable code, and the app runs them in the browser's WebAssembly sandbox.

## Scope

In scope: a model file that differs from its documented source, or that is crafted to misbehave in onnxruntime. Out of scope: the upstream models' accuracy, and issues in the GLP app itself (report those in its repository).
