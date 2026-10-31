"""Strict deterministic StarVLA PI0.5 websocket policy server.

This entry point is evaluation-only.  It preserves the native OpenPI PI0.5
model and preprocessing while disabling compiler autotuning and approximate or
non-deterministic CUDA execution paths that can change actions between runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket

# These must be set before importing torch or any StarVLA/OpenPI module.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import torch


STRICT_DETERMINISM_VERSION = "pytorch_eager_cublas_v1"


def _eager_compile(model=None, *args, **kwargs):
    """Process-local torch.compile replacement used by strict evaluation."""
    del args, kwargs
    if model is None:
        return lambda function: function
    return model


def configure_strict_determinism() -> None:
    """Configure PyTorch to fail closed on non-deterministic CUDA operations."""
    torch._dynamo.config.disable = True
    torch.use_deterministic_algorithms(True)
    # Every inference tensor is initialized by this path. Filling freshly
    # allocated storage is redundant; fixed-output and gradient checks cover
    # this switch while deterministic kernels remain enabled.
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def strict_metadata() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Strict PI0.5 policy server requires CUDA")
    capability = torch.cuda.get_device_capability(0)
    return {
        "strict_determinism": True,
        "determinism_version": STRICT_DETERMINISM_VERSION,
        "torch_compile_disabled": bool(torch._dynamo.config.disable),
        "torch_compile_policy": "identity_patch_v1",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_enabled": bool(
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": f"{capability[0]}.{capability[1]}",
    }


configure_strict_determinism()
# The upstream PI0.5 constructor assigns torch.compile(..., max-autotune) to
# sample_actions. Patch before importing model code so that assignment is
# unambiguously eager on every supported PyTorch version.
torch.compile = _eager_compile

# Import only after the process-wide contract is installed.
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--config_yaml", default=None)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    return parser


def main(args: argparse.Namespace) -> None:
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        config_yaml=args.config_yaml,
    )

    # The upstream PI0.5 constructor requests high matmul precision and wraps
    # sample_actions with torch.compile. Re-assert the strict global contract
    # after construction; TORCH_COMPILE_DISABLE makes that wrapper run eagerly.
    configure_strict_determinism()
    runtime = strict_metadata()
    expected = {
        "torch_compile_disabled": True,
        "torch_compile_policy": "identity_patch_v1",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "tf32_enabled": False,
        "float32_matmul_precision": "highest",
        "cudnn_benchmark": False,
    }
    mismatches = {key: (runtime.get(key), value) for key, value in expected.items() if runtime.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Strict deterministic runtime contract is not active: {mismatches}")

    metadata = dict(wrapper.metadata)
    metadata.update(runtime)
    hostname = socket.gethostname()
    logging.info(
        "Creating strict deterministic server (host=%s, ip=%s, metadata=%s)",
        hostname,
        socket.gethostbyname(hostname),
        metadata,
    )
    WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
