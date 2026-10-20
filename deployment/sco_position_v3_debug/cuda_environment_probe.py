"""Compare legacy/corrected CUDA loading in separate short-lived processes."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

SYSTEM_DRIVER_DIRS = ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu")


def child():
    result = {"library_path": os.environ.get("LD_LIBRARY_PATH", "")}
    try:
        cuda = ctypes.CDLL("libcuda.so.1")

        class DlInfo(ctypes.Structure):
            _fields_ = [("fname", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("sname", ctypes.c_char_p), ("saddr", ctypes.c_void_p)]

        dl = ctypes.CDLL("libdl.so.2")
        info = DlInfo()
        dl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(DlInfo))
        if dl.dladdr(ctypes.cast(cuda.cuInit, ctypes.c_void_p), ctypes.byref(info)):
            result["loaded_driver_library"] = info.fname.decode()
        version = ctypes.c_int()
        result["driver_version_status"] = cuda.cuDriverGetVersion(ctypes.byref(version))
        result["driver_api_version"] = version.value
        if "--resolve-only" not in sys.argv:
            import torch
            torch.cuda.init()
            result["torch"] = torch.__version__
            result["torch_cuda"] = torch.version.cuda
            result["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            value = torch.arange(16, device="cuda", dtype=torch.float32).sum().item()
            result["allocation_sum"] = value
            result["passed"] = len(result["devices"]) == 2 and all(
                "H100" in d for d in result["devices"]) and value == 120.0
        else:
            result["resolved"] = True
    except Exception as exc:
        result.update(passed=False, error=repr(exc))
    print(json.dumps(result), flush=True)


def main():
    if "--child" in sys.argv:
        child()
        return
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.resolve().relative_to(Path("/data/xiangyuyue/ULLM-zf/RobotVla"))
    env = dict(os.environ)
    legacy = ":".join(p for p in env.get("LD_LIBRARY_PATH", "").split(":")
                      if p and p not in SYSTEM_DRIVER_DIRS)
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    if "--resolve-only" in sys.argv:
        command.append("--resolve-only")
    results = {}
    for name, path in (("legacy", legacy), ("corrected", legacy + ":" + ":".join(SYSTEM_DRIVER_DIRS))):
        proc = subprocess.run(command, env={**env, "LD_LIBRARY_PATH": path},
                              capture_output=True, text=True, timeout=60)
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            result = {"passed": False, "stdout": proc.stdout[-3000:]}
        result.update(returncode=proc.returncode, stderr=proc.stderr[-3000:])
        results[name] = result
    results["passed"] = results["corrected"].get("passed", False)
    results["resolve_only"] = "--resolve-only" in sys.argv
    output.write_text(json.dumps(results, indent=2))
    print("CUDA_ENVIRONMENT_PROBE " + json.dumps(results), flush=True)
    if not results["resolve_only"] and not results["passed"]:
        raise RuntimeError("Corrected CUDA environment failed; preserve evidence and HOLD")


if __name__ == "__main__":
    main()
