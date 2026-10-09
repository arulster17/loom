"""Filesystem layout inside a Loom RunPod pod, and the pinned client runtime.

The pod scripts (`runpod_scripts/*.sh`) take these as rendered variables; keeping
them here gives the provider and the tests one source.
"""

from __future__ import annotations

LOOM_ROOT = "/var/lib/loom"
STAGE_FILE = f"{LOOM_ROOT}/stages"
CTL_ROOT = f"{LOOM_ROOT}/ctl"  # root-only; detached scripts, their URLs and output
JOBS_DIR = f"{LOOM_ROOT}/jobs"  # one job-user-owned work dir per run
JOB_HOME = f"{LOOM_ROOT}/home"
SSHD_CONFIG = f"{LOOM_ROOT}/sshd_config"
ENGINE_PIDFILE = f"{LOOM_ROOT}/engine.pid"
SSH_DIR = "/root/.ssh"
LOG_DIR = "/var/log/loom"
WEIGHTS_DIR = "/opt/loom/hf"  # HF_HOME for the engine; excluded from the secret scan
PYTHON_DIR = "/opt/loom/python"  # root-owned client interpreter
# Root-owned, world-readable copies of pinned workload datasets: <sha256>/<file name>.
DATA_DIR = "/opt/loom/data"
CLIENT_ENV_ROOT = "/opt/loom/clientenv"  # root-owned virtualenvs, one per wheel+requirements
PROC1_ENVIRON = "/proc/1/environ"
PROC_ROOT = "/proc"
# Where a secret value must never be found once the engine is up (weights excluded).
SECRET_SCAN_DIRS = ("/etc", "/opt/loom", LOOM_ROOT, "/tmp", "/root", LOG_DIR)

JOB_USER = "loom"
JOB_UID = 10001

# The bench client runs on a pinned python-build-standalone CPython, not the
# engine image's Python, so the client runtime is the same in every engine image.
# sha256 verified against the release's published digest on 2026-10-06.
CLIENT_PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20261003/"
    "cpython-3.12.15%2B20261003-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
)
CLIENT_PYTHON_SHA256 = "731af898886c5f821890dc901eca3c651cca8e51fa7308c159d12a1194aeac91"
