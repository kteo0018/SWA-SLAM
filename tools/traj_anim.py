#!/usr/bin/env python3
"""Demo video 3: each method's worst run (of 30) on the flight path where the
baseline fails hardest. Per-segment aligned, same protocol as the paper."""
import glob, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

EB = "/home/ken/sota_slam_compare/eval_batch_dedup"
S2 = "/home/ken/ORB-SLAM3-ROS2-Docker-Fork/s2_slam_results"
PATHS = {"gz_dyna_palm_p1r1_results": "Flight Path 1", "gz_dyna_palm_p2r1_results": "Flight Path 2",
         "gz_dyna_palm_p3r1_results": "Flight Path 3", "gz_dyna_palm_p5r1_results": "Flight Path 4"}
OUT = "/home/ken/swa_portfolio/03_trajectory_comparison.mp4"

def load_tum(p):
    ts, xyz = [], []
    for ln in open(p):
        f = ln.split()
        if len(f) < 8 or ln.startswith("#"): continue
        t = float(f[0])
        if ts and t == ts[-1]: continue
        ts.append(t); xyz.append([float(f[1]), float(f[2]), float(f[3])])
    return np.array(ts), np.array(xyz)

def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    sc, dc = src - ms, dst - md
    U, D, Vt = np.linalg.svd(dc.T @ sc / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((sc**2).sum() / len(src))
    return s, R, md - s * R @ ms

def prepare(traj_f, gt_t, gt_p):
    ts, xyz = load_tum(traj_f)
    idx = np.clip(np.searchsorted(gt_t, ts), 1, len(gt_t) - 1)
    idx[np.abs(gt_t[idx - 1] - ts) < np.abs(gt_t[idx] - ts)] -= 1
    ok = np.abs(gt_t[idx] - ts) < 0.05
    cut = np.where(np.diff(ts) > 0.5)[0] + 1
    aligned = np.full_like(xyz, np.nan)
    err = []
    for seg in np.split(np.arange(len(ts)), cut):
        seg = seg[ok[seg]]
        if len(seg) < 3: continue
        s, R, t = umeyama(xyz[seg], gt_p[idx[seg]])
        a = s * (R @ xyz[seg].T).T + t
        aligned[seg] = a
        err += list(np.linalg.norm(a - gt_p[idx[seg]], axis=1))
    return ts, aligned, float(np.sqrt(np.mean(np.square(err)))) if err else float("nan")

def worst(folder, pattern, gt_t, gt_p):
    best = None
    for f in glob.glob(os.path.join(folder, pattern)):
        ts, al, rmse = prepare(f, gt_t, gt_p)
        if len(ts) < 100 or np.isnan(rmse): continue
        if best is None or rmse > best[3]: best = (f, ts, al, rmse)
    return best

# pick the path where the baseline's worst run is worst
pick = None
for d, label in PATHS.items():
    gt_t, gt_p = load_tum(os.path.join(EB, d, "ground_truth.tum"))
    w = worst(os.path.join(EB, d), "orb_slam3_ori*.txt", gt_t, gt_p)
    print("%s baseline worst %.3f (%s)" % (label, w[3], os.path.basename(w[0])))
    if pick is None or w[3] > pick[2][3]: pick = (d, label, w, gt_t, gt_p)
d, plabel, wori, gt_t, gt_p = pick
wswa = worst(os.path.join(S2, d, "sensitivity_c30", "delta_j", "2p5"), "orb_slam3_c1-green-ring*.txt", gt_t, gt_p)
print("chosen:", plabel, "| baseline %.3f  swa %.3f" % (wori[3], wswa[3]))

runs = [("Baseline SLAM", "tab:red", wori[1], wori[2], wori[3]),
        ("SWA-SLAM (proposed)", "tab:green", wswa[1], wswa[2], wswa[3])]
t0, t1 = gt_t[0], gt_t[-1]
frames = np.linspace(t0, t1, 820)
mx = 0.06 * (gt_p[:, 0].max() - gt_p[:, 0].min())
my = 0.18 * (gt_p[:, 1].max() - gt_p[:, 1].min())
xlim = (gt_p[:, 0].min() - mx, gt_p[:, 0].max() + mx)
ylim = (gt_p[:, 1].min() - my, gt_p[:, 1].max() + my)
ext_w = gt_p[:, 0].max() - gt_p[:, 0].min()
ext_h = gt_p[:, 1].max() - gt_p[:, 1].min()
if ext_h > ext_w:   # tall path: panels side by side
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.4), dpi=112, constrained_layout=True)
else:               # wide path: panels stacked
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.6), dpi=112, constrained_layout=True)
fig.suptitle("Worst run of 30 for each method - simulated palm plantation, wind on (%s)" % plabel, fontsize=10)
arts = []
for ax, (name, col, ts, al, rmse) in zip(axes, runs):
    ax.plot(gt_p[:, 0], gt_p[:, 1], color="0.75", lw=1.2, label="ground truth")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title("%s - worst-run RMSE %.2f m" % (name, rmse), fontsize=10)
    ln, = ax.plot([], [], color=col, lw=1.6, label="estimate")
    dot, = ax.plot([], [], "o", color=col, ms=5)
    txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=8)
    lost = ax.text(0.5, 0.82, "", transform=ax.transAxes, fontsize=13, color="red", ha="center", weight="bold")
    arts.append((ln, dot, txt, lost, ts, al))
    ax.legend(loc="upper right", fontsize=7)

def update(tk):
    out = []
    for ln, dot, txt, lost, ts, al in arts:
        m = ts <= tk
        ln.set_data(al[m, 0], al[m, 1])
        if m.any() and np.isfinite(al[m][-1, 0]):
            dot.set_data([al[m][-1, 0]], [al[m][-1, 1]])
        recent = m.any() and (tk - ts[m][-1] < 0.3)
        lost.set_text("" if recent or tk < ts[0] or tk > ts[-1] + 0.3 else "TRACKING LOST")
        txt.set_text("t = %5.1f s" % (tk - t0))
        out += [ln, dot, txt, lost]
    return out

anim = FuncAnimation(fig, update, frames=frames, blit=True)
anim.save(OUT, writer=FFMpegWriter(fps=24, extra_args=["-pix_fmt", "yuv420p", "-crf", "23"]))
print("wrote", OUT)
