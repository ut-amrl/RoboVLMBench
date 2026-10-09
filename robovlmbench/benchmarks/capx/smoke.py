"""Smoke check of one CaP-X suite on this node. Runs in a CaP-X env (Python 3.10 or 3.12), never in the main env:
    source scripts/capx_env.sh robosuite && cd "$CAPX_DIR" && python <this file> robosuite

Run by scripts/check_env.py. Builds one task from CaP-X's own human-written oracle configs, renders the camera
headless (EGL on the GPU), and runs the oracle's code through the same sandbox the model's code goes through:
    robosuite  cube lifting, privileged API (needs CaP-X's PyRoKi IK server)
    libero     LIBERO-PRO libero_object_swap task 7, privileged API (a perturbed suite: its files are in place)
Prints one JSON line with the result; exits non-zero unless the image is not blank and the task gets done.
"""

import json
import os
import socket
import sys
import time

# This file's folder holds the harness's other modules; keep their names out of CaP-X's imports.
if os.path.dirname(os.path.abspath(__file__)) in sys.path:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))

ORACLE_CONFIGS = {
    "robosuite": "env_configs/human_oracle_code/robosuite/franka_robosuite_cube_lifting_privileged_oracle.yaml",
    "libero": "env_configs/human_oracle_code/libero/franka_libero_object_swap_7_privileged_oracle.yaml",
}


def main(suite: str) -> None:
    import numpy as np

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader
    from capx.envs.runner import _start_api_servers, _stop_api_servers

    started = time.monotonic()
    config = DictLoader.load([ORACLE_CONFIGS[suite]])
    servers = config.get("api_servers") or []
    busy = [s["port"] for s in servers if _port_open(s["port"])]
    if busy:  # exit code 3: not a problem of this hardware
        print(f"ports {busy} are in use (a CaP-X run on this node?); CaP-X's servers have fixed ports", file=sys.stderr)
        sys.exit(3)
    processes = _start_api_servers(servers)
    try:
        env = instantiate(config["env"])
        env.reset(seed=1, options={"trial": 1})
        image = np.asarray(env.render())
        _, reward, _, _, info = env.step(env.oracle_code)
    finally:
        _stop_api_servers(processes)
    result = {"suite": suite, "image": list(image.shape), "image_std": round(float(image.std()), 1),
              "task_completed": bool(info.get("task_completed")), "reward": float(reward),
              "seconds": round(time.monotonic() - started)}
    print(json.dumps(result))
    if image.ndim != 3 or image.std() < 5:
        sys.exit("the rendered image is blank")
    if not result["task_completed"]:
        sys.exit(f"the oracle's code did not get the task done; stderr:\n{info.get('stderr', '')[-2000:]}")


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", int(port))) == 0


if __name__ == "__main__":
    main(sys.argv[1])
