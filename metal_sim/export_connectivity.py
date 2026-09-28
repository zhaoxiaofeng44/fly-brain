"""Export fly-brain connectivity to binary files for the Metal simulator.

Units: everything in mV / ms (float32), matching the Brian2 model params.
Outputs (in ~/fly-brain/metal_sim/data/):
  meta.json          n_neurons, n_syn, model params, group definitions
  csr_start.i32      n+1   CSR row offsets (by presynaptic neuron)
  csr_post.i32       n_syn postsynaptic indices
  csr_w.f32          n_syn weights in mV (Excitatory x Connectivity * 0.275)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
BRAIN = Path.home() / "fly-brain" / "data"

W_SYN_MV = 0.275
GROUPS = {
    "sugar": [720575940624963786, 720575940630233916, 720575940637568838,
              720575940638202345, 720575940617000768, 720575940630797113,
              720575940632889389, 720575940621754367, 720575940621502051,
              720575940640649691, 720575940639332736, 720575940616885538,
              720575940639198653, 720575940639259967, 720575940617937543,
              720575940632425919, 720575940633143833, 720575940612670570,
              720575940628853239, 720575940629176663, 720575940611875570],
    "p9_left": [720575940627652358],
    "p9_right": [720575940635872101],
}

comp = pd.read_csv(BRAIN / "2025_Completeness_783.csv", index_col=0)
n = len(comp)
flyid2i = {j: i for i, j in enumerate(comp.index)}

con = pd.read_parquet(BRAIN / "2025_Connectivity_783.parquet")
pre = con["Presynaptic_Index"].to_numpy(dtype=np.int32)
post = con["Postsynaptic_Index"].to_numpy(dtype=np.int32)
w = (con["Excitatory x Connectivity"].to_numpy(dtype=np.float32) * W_SYN_MV).astype(np.float32)
del con
print(f"n_neurons={n} n_syn={len(pre)}")

order = np.argsort(pre, kind="stable")
post_csr = post[order]
w_csr = w[order]
counts = np.bincount(pre, minlength=n)
csr_start = np.zeros(n + 1, dtype=np.int32)
np.cumsum(counts, out=csr_start[1:])

csr_start.tofile(DATA / "csr_start.i32")
post_csr.tofile(DATA / "csr_post.i32")
w_csr.tofile(DATA / "csr_w.f32")

groups_idx = {name: [flyid2i[fid] for fid in ids] for name, ids in GROUPS.items()}
for name, idxs in groups_idx.items():
    arr = np.array([len(idxs)] + list(idxs), dtype=np.int32)
    arr.tofile(DATA / f"group_{name}.i32")
meta = {
    "n_neurons": int(n),
    "n_syn": int(len(pre)),
    "dt_ms": 0.1,
    "v0_mv": -52.0, "vr_mv": -52.0, "th_mv": -45.0,
    "t_mbr_ms": 20.0, "tau_ms": 5.0,
    "rfc_steps": 22, "dly_steps": 18,
    "w_poi_mv": W_SYN_MV * 250.0,
    "groups": groups_idx,
}
(DATA / "meta.json").write_text(json.dumps(meta, indent=1))
print("exported to", DATA)
