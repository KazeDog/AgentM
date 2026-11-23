"""Process-local guard for the provided offline checks (not an OS sandbox)."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NETWORK_ATTEMPTS: list[str] = []


def activate() -> None:
    # Do not let the check discover credentials inherited from the workstation.
    for name in list(os.environ):
        if any(x in name.upper() for x in ("API_KEY", "API_TOKEN", "ACCESS_TOKEN")):
            os.environ.pop(name, None)
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "AGENTM_ENABLE_NETWORK_LOOKUP": "0", "WANDB_MODE": "disabled",
        "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false",
    })
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="agentm-mpl-")
    os.environ["MPLBACKEND"] = "Agg"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    def audit(event, args):
        if event in {"socket.connect", "socket.sendto", "socket.getaddrinfo"}:
            NETWORK_ATTEMPTS.append(event)
            raise RuntimeError("Offline check blocked network operation: " + event)
        # Matplotlib probes fonts and NumPy probes CPU capabilities.
        if event == "subprocess.Popen" and str(args[0]) in {"fc-list", "fc-match", "lscpu"}:
            return
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn"}:
            raise RuntimeError("Offline check blocked child-process execution")

    sys.addaudithook(audit)


def verify_import_roots() -> dict[str, str]:
    import agentm
    import agentm.runtime
    result = {}
    for module in (agentm, agentm.runtime):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError("Import escaped the submission directory: " + module.__name__)
        result[module.__name__] = str(path.relative_to(ROOT))
    return result
