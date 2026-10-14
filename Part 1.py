import argparse
import os

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# 1. Detection + description (existing SIFT implementation)
# ----------------------------------------------------------------------------
def load_image(path):
    """Return (color BGR image, grayscale uint8 image)."""
    color = cv2.imread(path, cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, gray


def detect_sift(gray, n_features=0, n_octave_layers=3, contrast_threshold=0.04,
                edge_threshold=10, sigma=1.6):
    sift = cv2.SIFT_create(nfeatures=n_features,
                           nOctaveLayers=n_octave_layers,
                           contrastThreshold=contrast_threshold,
                           edgeThreshold=edge_threshold,
                           sigma=sigma)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        desc = np.zeros((0, 128), dtype=np.float32)
    return kps, desc


# ----------------------------------------------------------------------------
# 2. Matching from first principles
# ----------------------------------------------------------------------------
def euclidean_distances(A, B, chunk=1024):
    
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    b_sq = np.sum(B * B, axis=1)                      
    out = np.empty((A.shape[0], B.shape[0]), dtype=np.float32)
    for s in range(0, A.shape[0], chunk):
        a = A[s:s + chunk]
        a_sq = np.sum(a * a, axis=1)[:, None]        
        d2 = a_sq + b_sq[None, :] - 2.0 * (a @ B.T)   
        np.maximum(d2, 0.0, out=d2)                   
        out[s:s + chunk] = np.sqrt(d2)
    return out


def two_nearest(dist):
    """
    For each row of the distance matrix, return indices and distances of the
    nearest and second-nearest columns.
    Returns idx1, idx2, d1, d2 (each length n1), with d1 <= d2.
    """
    if dist.shape[1] < 2:
        raise ValueError("Need at least two descriptors in the second image.")
    # argpartition finds the 2 smallest in O(n) per row; then order those two.
    part = np.argpartition(dist, 1, axis=1)[:, :2]
    rows = np.arange(dist.shape[0])[:, None]
    two = dist[rows, part]
    order = np.argsort(two, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    two = np.take_along_axis(two, order, axis=1)
    return part[:, 0], part[:, 1], two[:, 0], two[:, 1]


def ratio_test_matches(desc1, desc2, ratio=0.75, mutual=False, dist=None):
    if len(desc1) == 0 or len(desc2) < 2:
        empty = np.array([], dtype=int)
        e = np.array([], dtype=float)
        return {"i": empty, "j": empty, "d1": e, "d2": e, "ratio": e}

    if dist is None:
        dist = euclidean_distances(desc1, desc2)
    nn1, _, d1, d2 = two_nearest(dist)

    # d2 == 0 means duplicate descriptors; treat the ratio as 1 (reject).
    r = np.divide(d1, d2, out=np.ones_like(d1), where=d2 > 0)
    keep = r < ratio

    if mutual:
        back_nn = np.argmin(dist, axis=0)             # best image-1 match for each image-2 descriptor
        keep &= back_nn[nn1] == np.arange(len(desc1))

    i = np.nonzero(keep)[0]
    return {"i": i, "j": nn1[i], "d1": d1[i], "d2": d2[i], "ratio": r[i]}


def matches_to_points(kps1, kps2, m):
    """Convert matches to two (N, 2) arrays of (x, y) pixel coordinates."""
    p1 = np.array([kps1[i].pt for i in m["i"]], dtype=np.float64).reshape(-1, 2)
    p2 = np.array([kps2[j].pt for j in m["j"]], dtype=np.float64).reshape(-1, 2)
    return p1, p2


# ----------------------------------------------------------------------------
# 3. Visualization
# ----------------------------------------------------------------------------
def draw_matches(img1, img2, pts1, pts2, colors=None, title=None, max_lines=None, seed=0):
    """
    Side-by-side image with a line per match. `colors` is an optional (N, 3) BGR array
    (Part 3 can reuse this to color inliers green and outliers red).
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    n = len(pts1)
    idx = np.arange(n)
    if max_lines is not None and n > max_lines:
        idx = np.random.default_rng(seed).choice(n, max_lines, replace=False)

    thickness = max(1, int(round(min(canvas.shape[:2]) / 600)))
    radius = 3 * thickness
    for k in idx:
        c = (0, 255, 0) if colors is None else tuple(int(v) for v in colors[k])
        a = (int(round(pts1[k][0])), int(round(pts1[k][1])))
        b = (int(round(pts2[k][0])) + w1, int(round(pts2[k][1])))
        cv2.line(canvas, a, b, c, thickness, cv2.LINE_AA)
        cv2.circle(canvas, a, radius, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, radius, c, -1, cv2.LINE_AA)

    if title:
        cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def draw_keypoints(img, kps):
    return cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# ----------------------------------------------------------------------------
# 4. Experiments for the README
# ----------------------------------------------------------------------------
def run_single(color1, gray1, color2, gray2, ratio, mutual, sift_kwargs, out_dir, tag):
    kps1, d1 = detect_sift(gray1, **sift_kwargs)
    kps2, d2 = detect_sift(gray2, **sift_kwargs)
    m = ratio_test_matches(d1, d2, ratio=ratio, mutual=mutual)
    p1, p2 = matches_to_points(kps1, kps2, m)
    title = f"ratio={ratio}  matches={len(p1)}"
    vis = draw_matches(color1, color2, p1, p2, title=title)
    path = os.path.join(out_dir, f"matches_{tag}.jpg")
    cv2.imwrite(path, vis)
    print(f"[{tag}] kps: {len(kps1)} / {len(kps2)}   matches kept: {len(p1)}   -> {path}")
    return kps1, kps2, m


def sweep_ratio(color1, gray1, color2, gray2, ratios, mutual, sift_kwargs, out_dir):
    """Same image pair, several ratio thresholds. Descriptors and distances computed once."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kps1, desc1 = detect_sift(gray1, **sift_kwargs)
    kps2, desc2 = detect_sift(gray2, **sift_kwargs)
    print(f"Keypoints: {len(kps1)} in image 1, {len(kps2)} in image 2")
    dist = euclidean_distances(desc1, desc2)

    counts = []
    for r in ratios:
        m = ratio_test_matches(desc1, desc2, ratio=r, mutual=mutual, dist=dist)
        p1, p2 = matches_to_points(kps1, kps2, m)
        counts.append(len(p1))
        vis = draw_matches(color1, color2, p1, p2, title=f"ratio={r}  matches={len(p1)}")
        path = os.path.join(out_dir, f"matches_ratio_{r:.2f}.jpg")
        cv2.imwrite(path, vis)
        print(f"ratio={r:.2f}  matches kept={len(p1):5d}  -> {path}")

    plt.figure(figsize=(5, 3.5))
    plt.plot(ratios, counts, "o-")
    plt.xlabel("ratio threshold r")
    plt.ylabel("tentative matches kept")
    plt.title("Matches retained vs. ratio threshold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "ratio_vs_count.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")


def sweep_sift(color1, gray1, color2, gray2, ratio, mutual, out_dir):
    """Vary one SIFT parameter at a time and report keypoints and matches at a fixed ratio."""
    base = dict(n_features=0, n_octave_layers=3, contrast_threshold=0.04,
                edge_threshold=10, sigma=1.6)
    grid = {
        "contrast_threshold": [0.01, 0.02, 0.04, 0.08],
        "edge_threshold": [5, 10, 20],
        "n_features": [500, 1000, 2000, 0],
        "sigma": [1.2, 1.6, 2.0],
    }
    rows = []
    for name, values in grid.items():
        for v in values:
            kw = dict(base)
            kw[name] = v
            k1, d1 = detect_sift(gray1, **kw)
            k2, d2 = detect_sift(gray2, **kw)
            m = ratio_test_matches(d1, d2, ratio=ratio, mutual=mutual)
            rows.append((name, v, len(k1), len(k2), len(m["i"])))

    header = f"{'parameter':20s} {'value':>8s} {'kps1':>7s} {'kps2':>7s} {'matches':>8s}"
    lines = [header, "-" * len(header)]
    for name, v, a, b, c in rows:
        lines.append(f"{name:20s} {v!s:>8s} {a:7d} {b:7d} {c:8d}")
    table = "\n".join(lines)
    print(table)
    path = os.path.join(out_dir, "sift_param_sweep.txt")
    with open(path, "w") as f:
        f.write(f"ratio = {ratio}\n\n{table}\n")
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser(description="Part 1: SIFT + ratio-test feature matching")
    ap.add_argument("img1")
    ap.add_argument("img2")
    ap.add_argument("--ratio", type=float, default=0.75, help="ratio threshold r (default 0.75)")
    ap.add_argument("--mutual", action="store_true", help="also require mutual nearest neighbors")
    ap.add_argument("--sweep-ratio", type=float, nargs="+", metavar="R",
                    help="compare several ratio thresholds on this pair")
    ap.add_argument("--sweep-sift", action="store_true", help="vary SIFT parameters")
    ap.add_argument("--max-width", type=int, default=1200,
                    help="downscale images wider than this before SIFT (0 = no resize)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-features", type=int, default=0)
    ap.add_argument("--n-octave-layers", type=int, default=3)
    ap.add_argument("--contrast-threshold", type=float, default=0.04)
    ap.add_argument("--edge-threshold", type=float, default=10)
    ap.add_argument("--sigma", type=float, default=1.6)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    color1, gray1 = load_image(args.img1)
    color2, gray2 = load_image(args.img2)

    if args.max_width:
        def shrink(c, g):
            if c.shape[1] <= args.max_width:
                return c, g
            s = args.max_width / c.shape[1]
            c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            return c, cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        color1, gray1 = shrink(color1, gray1)
        color2, gray2 = shrink(color2, gray2)

    sift_kwargs = dict(n_features=args.n_features, n_octave_layers=args.n_octave_layers,
                       contrast_threshold=args.contrast_threshold,
                       edge_threshold=args.edge_threshold, sigma=args.sigma)

    if args.sweep_ratio:
        sweep_ratio(color1, gray1, color2, gray2, args.sweep_ratio, args.mutual, sift_kwargs, args.out)
    elif args.sweep_sift:
        sweep_sift(color1, gray1, color2, gray2, args.ratio, args.mutual, args.out)
    else:
        run_single(color1, gray1, color2, gray2, args.ratio, args.mutual, sift_kwargs,
                   args.out, f"ratio_{args.ratio:.2f}")


if __name__ == "__main__":
    main()