"""Check that every model in a release directory is what SHA256SUMS says and
is a plain, self-contained ONNX graph.

ONNX is a protobuf graph description, not executable code (unlike pickle
based formats), so the realistic risks are a swapped file, a reference to
external data on disk, or an operator from a custom domain that would need
native code. Each of those fails this check, and the model must load in
onnxruntime.

Usage: python scripts/validate.py <dir-with-onnx-and-SHA256SUMS>
"""

import collections
import hashlib
import pathlib
import sys

import onnx
import onnx.checker
import onnxruntime

ALLOWED_DOMAINS = {"", "ai.onnx", "com.microsoft"}
# Contrib ops onnxruntime's own dynamic quantizer and fusions emit.
ALLOWED_MS_OPS = {"DynamicQuantizeMatMul", "MatMulIntegerToFloat", "FusedMatMul", "QuickGelu", "Gelu"}
MAX_BYTES = 100 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def graphs(g):
    yield g
    for node in g.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from graphs(attr.g)
            for sub in attr.graphs:
                yield from graphs(sub)


def tensors(g):
    yield from g.initializer
    for node in g.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors


def check_model(path):
    errors = []
    if path.stat().st_size > MAX_BYTES:
        errors.append(f"larger than {MAX_BYTES} bytes")
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    if model.functions:
        errors.append("contains local functions")
    if len(model.training_info):
        errors.append("contains training info")
    for imp in model.opset_import:
        if imp.domain not in ALLOWED_DOMAINS:
            errors.append(f"opset domain {imp.domain!r} not allowed")
    ops = collections.Counter()
    for g in graphs(model.graph):
        for node in g.node:
            ops[(node.domain or "", node.op_type)] += 1
            if node.domain not in ALLOWED_DOMAINS:
                errors.append(f"node {node.name!r} uses domain {node.domain!r}")
            elif node.domain == "com.microsoft" and node.op_type not in ALLOWED_MS_OPS:
                errors.append(f"contrib op {node.op_type!r} not on the allowlist")
            elif node.domain in ("", "ai.onnx") and not onnx.defs.has(node.op_type):
                errors.append(f"unknown op {node.op_type!r}")
        for t in tensors(g):
            if t.data_location == onnx.TensorProto.EXTERNAL or len(t.external_data):
                errors.append(f"tensor {t.name!r} references external data")
    if not errors:
        onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return errors, ops


def main(root):
    root = pathlib.Path(root)
    sums = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*")] = digest
    failed = False
    for name, digest in sorted(sums.items()):
        path = root / name
        if sha256(path) != digest:
            print(f"FAIL {name}: SHA-256 mismatch")
            failed = True
            continue
        errors, ops = check_model(path)
        summary = ", ".join(f"{d + '.' if d else ''}{o}" for d, o in sorted(ops))
        print(f"{'FAIL' if errors else 'OK  '} {name}: {sum(ops.values())} nodes; ops: {summary}")
        for e in errors:
            print(f"     - {e}")
        failed |= bool(errors)
    onnx_files = {p.name for p in root.glob("*.onnx")}
    if onnx_files - sums.keys():
        print(f"FAIL unlisted files: {sorted(onnx_files - sums.keys())}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
