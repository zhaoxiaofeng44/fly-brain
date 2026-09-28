"""Python bridge to the metal_sim GPU brain simulator."""
import json
import subprocess
from pathlib import Path

import numpy as np


class MetalBrain:
    def __init__(self, data_dir: Path | None = None):
        root = Path.home() / "fly-brain" / "metal_sim"
        self.data_dir = Path(data_dir) if data_dir else root / "data"
        sim = root / "metal_sim"
        self.p = subprocess.Popen(
            [str(sim), str(self.data_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        ready = self.p.stdout.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"sim did not start: {ready}")
        parts = ready.split()
        self.n = int(parts[1][2:])
        self.n_syn = int(parts[2][4:])
        meta = json.loads((self.data_dir / "meta.json").read_text())
        self.groups = {k: np.array(v, dtype=np.int64)
                       for k, v in meta["groups"].items()}
        self.dt_ms = meta["dt_ms"]
        self.total_steps = 0

    def _readline(self) -> str:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("sim died")
        return line.strip()

    def cmd(self, c: str) -> str:
        self.p.stdin.write(c + "\n")
        self.p.stdin.flush()
        return self._readline()

    def set_rate(self, group: str, hz: float) -> None:
        out = self.cmd(f"rate {group} {hz}")
        if not out.startswith("OK"):
            raise RuntimeError(out)

    def run(self, seconds: float | None = None, steps: int | None = None) -> None:
        n = steps if steps is not None else int(round(seconds * 1000 / self.dt_ms))
        out = self.cmd(f"run {n}")
        self.total_steps = int(out.split()[1])

    def reset(self) -> None:
        self.cmd("reset")
        self.total_steps = 0

    def drain_spikes(self) -> tuple[np.ndarray, np.ndarray]:
        self.p.stdin.write("spikes\n")
        self.p.stdin.flush()
        steps, neurons = [], []
        while True:
            line = self._readline()
            if line.startswith("E "):
                break
            _, s, n = line.split()
            steps.append(int(s))
            neurons.append(int(n))
        return np.array(steps, dtype=np.int64), np.array(neurons, dtype=np.int64)

    def close(self) -> None:
        try:
            self.cmd("quit")
        except Exception:
            pass
        self.p.wait(timeout=5)
