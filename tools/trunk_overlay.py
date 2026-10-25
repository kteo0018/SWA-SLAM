#!/usr/bin/env python3
"""Demo video 2: trunk detector overlay (paper parameters) on the p1r1 camera feed."""
import cv2, glob, numpy as np

RGB = sorted(glob.glob("/home/ken/ORB-SLAM3-ROS2-Docker-Fork/s1_gz_dataset/gz_dyna_palm_p1r1_tum/rgb/*.png"))[::2]
OUT = "/home/ken/swa_portfolio/02_trunk_detection.mp4"
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K2 = np.ones((2, 2), np.uint8)

def detect(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 60, 20), (85, 255, 150))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, K3)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cand = cv2.bitwise_and(cv2.bitwise_not(green), (gray > 25).astype(np.uint8) * 255)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, K5)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, K2)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cand)
    H, W = gray.shape
    yy, xx = np.mgrid[0:H, 0:W]
    trunks = []
    for i in range(1, n):
        A = stats[i, cv2.CC_STAT_AREA]
        if not 60 <= A <= 3500: continue
        r = np.sqrt(A / np.pi)
        if not 4 <= r <= 15: continue
        m = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: continue
        L = cv2.arcLength(cnts[0], True)
        if L <= 0 or 4 * np.pi * A / L**2 < 0.18: continue
        cx, cy = cents[i]
        x0, x1 = max(0, int(cx - 4)), min(W, int(cx + 5))
        y0, y1 = max(0, int(cy - 4)), min(H, int(cy + 5))
        if green[y0:y1, x0:x1].mean() / 255.0 > 0.25: continue
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ring = (d2 >= (1.3 * r) ** 2) & (d2 <= (3.5 * r) ** 2)
        if ring.sum() == 0 or green[ring].mean() / 255.0 < 0.95: continue
        trunks.append((cx, cy, r))
    return green, trunks

first = cv2.imread(RGB[0]); H, W = first.shape[:2]
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for k, f in enumerate(RGB):
    bgr = cv2.imread(f)
    green, trunks = detect(bgr)
    vis = bgr.copy()
    vis[green > 0] = (0.75 * vis[green > 0] + 0.25 * np.array([0, 160, 0])).astype(np.uint8)
    for cx, cy, r in trunks:
        c = (int(cx), int(cy))
        cv2.circle(vis, c, max(2, int(r)), (0, 0, 255), 2)          # trunk (TC)
        cv2.circle(vis, c, int(2.5 * r), (0, 255, 255), 1)          # NT bound 2.5r
        cv2.circle(vis, c, int(3.5 * r), (255, 200, 0), 1)          # green-ring check radius
    cv2.rectangle(vis, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(vis, f"SWA-SLAM trunk extractor  |  trunks: {len(trunks)}  |  red=trunk  yellow=near-trunk bound (2.5r)  cyan=green-ring check (3.5r)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    vw.write(vis)
vw.release()
print("wrote", OUT, len(RGB), "frames")
