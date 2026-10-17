
import argparse
import sys
import cv2
import numpy as np


def detect(img, **sift_params):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.SIFT_create(**sift_params).detectAndCompute(gray, None)


def match(desc1, desc2, r=0.7):
    
    sq = (desc1 ** 2).sum(1)[:, None] + (desc2 ** 2).sum(1)[None, :] - 2 * desc1 @ desc2.T
    dist = np.sqrt(np.maximum(sq, 0))

    nn = np.argsort(dist, axis=1)[:, :2]            
    rows = np.arange(len(desc1))
    d1, d2 = dist[rows, nn[:, 0]], dist[rows, nn[:, 1]]

    keep = d1 < r * d2                              
    return [cv2.DMatch(int(i), int(nn[i, 0]), float(d1[i])) for i in np.where(keep)[0]]


def show_matches(img1, kp1, img2, kp2, matches, path):
    vis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(path, vis)


# PArt 2

def matched_points(kp1, kp2, matches):
    
    src = np.array([kp2[m.trainIdx].pt for m in matches])
    dst = np.array([kp1[m.queryIdx].pt for m in matches])
    return src, dst


def apply_affine(points, H):
    num_points = len(points)
    homogeneous_points = np.zeros((3, num_points))
    for i in range(num_points):
        homogeneous_points[0, i] = points[i, 0]
        homogeneous_points[1, i] = points[i, 1]
        homogeneous_points[2, i] = 1
    transformed = H @ homogeneous_points
    return transformed[:2, :].T


def apply_homography(points, H):
    num_points = len(points)
    homogeneous_points = np.zeros((3, num_points))
    for i in range(num_points):
        homogeneous_points[0, i] = points[i, 0]
        homogeneous_points[1, i] = points[i, 1]
        homogeneous_points[2, i] = 1
    transformed = H @ homogeneous_points
    new_points = np.zeros((num_points, 2))
    for i in range(num_points):
        new_points[i, 0] = transformed[0, i] / transformed[2, i]  
        new_points[i, 1] = transformed[1, i] / transformed[2, i]   
    return new_points


def estimate_affine(src, dst):
    n = len(src)
    A, b = np.zeros((2 * n, 6)), np.zeros(2 * n)
    for i in range(n):
        x, y = src[i]
        xp, yp = dst[i]
        A[2 * i] = [x, y, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i], b[2 * i + 1] = xp, yp
    t = np.linalg.solve(A.T @ A, A.T @ b)          
    return np.vstack([t.reshape(2, 3), [0, 0, 1]])


def estimate_homography(src, dst):
    
    n = len(src)
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src[i]
        xp, yp = dst[i]
        A[2 * i] = [x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp]
    eigvals, eigvecs = np.linalg.eigh(A.T @ A)      
    H = eigvecs[:, 0].reshape(3, 3)                 
    return H / H[2, 2]                              


# Part 3
MODELS = {"affine": (estimate_affine, apply_affine, 3),
          "homography": (estimate_homography, apply_homography, 4)}


def reproj_error(src, dst, H, model):
    
    return np.linalg.norm(MODELS[model][1](src, H) - dst, axis=1)


def num_rounds(outlier_ratio, s, p=0.99):
    
    return int(np.ceil(np.log(1 - p) / np.log(1 - (1 - outlier_ratio) ** s)))


def ransac(src, dst, model="homography", n_iters=1000, thresh=3.0, seed=None):
   
    estimate, apply, s = MODELS[model]
    rng = np.random.default_rng(seed)
    best = np.zeros(len(src), dtype=bool)

    with np.errstate(all="ignore"):               
        for _ in range(n_iters):
            idx = rng.choice(len(src), s, replace=False)        
            try:
                H = estimate(src[idx], dst[idx])                
            except np.linalg.LinAlgError:                       
                continue
            inliers = np.linalg.norm(apply(src, H) - dst, axis=1) < thresh   
            if inliers.sum() > best.sum():                      
                best = inliers

    if best.sum() < s:
        raise ValueError("RANSAC found too few inliers; try a larger threshold or more matches")
    return estimate(src[best], dst[best]), best  


def show_inliers(img1, kp1, img2, kp2, matches, inliers, path):
    
    h = max(img1.shape[0], img2.shape[0])
    pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)))
    vis = np.hstack([pad(img1), pad(img2)])
    for m, ok in zip(matches, inliers):
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        cv2.line(vis, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)),
                 (0, 255, 0) if ok else (0, 0, 255), 1)
    cv2.imwrite(path, vis)


#Part 4-5

def warp_to_canvas(img, M, size):
    
    w, h = size
    M = M / (M @ [img.shape[1] / 2, img.shape[0] / 2, 1])[2]    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.linalg.inv(M) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])   
    with np.errstate(all="ignore"):
        x = (pts[0] / pts[2]).reshape(h, w)
        y = (pts[1] / pts[2]).reshape(h, w)
    inside = (x >= 0) & (x <= img.shape[1] - 1) & (y >= 0) & (y <= img.shape[0] - 1) & (pts[2] > 0).reshape(h, w)
    x, y = np.where(inside, x, -1).astype(np.float32), np.where(inside, y, -1).astype(np.float32)
    return cv2.remap(img, x, y, cv2.INTER_LINEAR), inside


def feather_weight(inside, power=1.0):
    
    d = cv2.distanceTransform(np.pad(inside.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    return d ** power


def warp_and_blend(images, Ms, power=1.0, max_pixels=100e6):

    corners = []
    for img, M in zip(images, Ms):
        h, w = img.shape[:2]
        corners.append(apply_homography(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float), M))
    corners = np.vstack(corners)

    x0, y0 = np.floor(corners.min(0))
    x1, y1 = np.ceil(corners.max(0))
    size = (int(x1 - x0) + 1, int(y1 - y0) + 1)                 

    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])      
    acc = np.zeros((size[1], size[0], 3), np.float32)
    total = np.zeros((size[1], size[0]), np.float32)
    for img, M in zip(images, Ms):
        warped, inside = warp_to_canvas(img, shift @ M, size)
        wgt = feather_weight(inside, power)
        acc += warped * wgt[..., None]
        total += wgt
    return (acc / np.maximum(total, 1e-6)[..., None]).astype(np.uint8)


def pairwise_transforms(images, model="homography", ratio=0.75, iters=1000, thresh=3.0, prefix="pair"):
    
    s = MODELS[model][2]
    feats = [detect(im) for im in images]
    H_list = []
    for i in range(len(images) - 1):
        (kp1, d1), (kp2, d2) = feats[i], feats[i + 1]
        matches = match(d1, d2, ratio)
        if len(matches) < s:
            raise ValueError(f"only {len(matches)} matches between images {i + 1} and {i + 2}; they may not overlap")
        src, dst = matched_points(kp1, kp2, matches)
        H, inliers = ransac(src, dst, model, iters, thresh)
        err = np.median(reproj_error(src[inliers], dst[inliers], H, model))
        print(f"images {i + 1}->{i + 2}: {len(matches)} matches, {inliers.sum()} inliers "
              f"({inliers.mean():.0%}), median inlier error {err:.2f} px")
        show_inliers(images[i], kp1, images[i + 1], kp2, matches, inliers, f"{prefix}{i + 1}_inliers.png")
        H_list.append(H)
    return H_list


def compose_to_reference(H_list, ref):
   
    to_ref = [None] * (len(H_list) + 1)
    to_ref[ref] = np.eye(3)
    for i in range(ref + 1, len(to_ref)):          
        to_ref[i] = to_ref[i - 1] @ H_list[i - 1]
    for i in range(ref - 1, -1, -1):               
        to_ref[i] = to_ref[i + 1] @ np.linalg.inv(H_list[i])
    return [M / M[2, 2] for M in to_ref]


def stitch(images, model="homography", ratio=0.75, iters=1000, thresh=3.0, power=1.0):
    
    H_list = pairwise_transforms(images, model, ratio, iters, thresh)
    ref = (len(images) - 1) // 2                  
    print(f"reference image: {ref + 1}")
    return warp_and_blend(images, compose_to_reference(H_list, ref), power)


def stitch_cli(argv):
    ap = argparse.ArgumentParser(prog="part1_matching.py stitch")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", choices=list(MODELS), default="homography")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--thresh", type=float, default=3.0)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--max_width", type=int, default=1000, help="downscale wider images (0 = off)")
    ap.add_argument("--out", default="panorama.png")
    args = ap.parse_args(argv)
    if len(args.images) < 2:
        ap.error("need at least two images")

    images = [cv2.imread(f) for f in args.images]
    if args.max_width:
        images = [cv2.resize(im, None, fx=args.max_width / im.shape[1], fy=args.max_width / im.shape[1],
                             interpolation=cv2.INTER_AREA) if im.shape[1] > args.max_width else im for im in images]
    cv2.imwrite(args.out, stitch(images, args.model, args.ratio, args.iters, args.thresh, args.power))
    print(f"saved {args.out}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["stitch"]:
        stitch_cli(sys.argv[2:])
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("img1")
    ap.add_argument("img2")
    ap.add_argument("ratios", type=float, nargs="*", default=[0.6, 0.8])
    ap.add_argument("--model", choices=list(MODELS), default="homography")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--thresh", type=float, default=3.0)
    args = ap.parse_args()

    img1, img2 = cv2.imread(args.img1), cv2.imread(args.img2)
    kp1, desc1 = detect(img1)
    kp2, desc2 = detect(img2)
    print(f"keypoints: {len(kp1)} in image 1, {len(kp2)} in image 2")

    estimate, _, s = MODELS[args.model]
    for r in args.ratios:
        matches = match(desc1, desc2, r)
        print(f"r = {r}: {len(matches)} tentative matches")
        show_matches(img1, kp1, img2, kp2, matches, f"matches_r{r}.png")
        if len(matches) < s:
            print(f"  need at least {s} matches for {args.model}, skipping RANSAC")
            continue

        src, dst = matched_points(kp1, kp2, matches)
        H_all = estimate(src, dst)                      
        H, inliers = ransac(src, dst, args.model, args.iters, args.thresh)
        err_before = reproj_error(src, dst, H_all, args.model)
        err_after = reproj_error(src[inliers], dst[inliers], H, args.model)

        print(f"  {args.model} RANSAC: {inliers.sum()} / {len(matches)} inliers ({inliers.mean():.0%})")
        print(f"  median error before RANSAC (fit and measured on all tentative matches): {np.median(err_before):.2f} px")
        print(f"  median error after RANSAC (final fit, measured on inliers only): {np.median(err_after):.2f} px")
        show_inliers(img1, kp1, img2, kp2, matches, inliers, f"ransac_{args.model}_r{r}.png")