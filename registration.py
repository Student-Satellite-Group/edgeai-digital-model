"""
registration.py — Task 4.1: ORB detection -> feature matching -> RANSAC
homography -> warp registration pipeline.

Registers a "moving" image (e.g. the thermal band) onto a "reference" image
(e.g. the optical band) so the pair can be fused pixel-aligned (Task 4.5).

Pipeline (all four steps chained by register_images()):
    detect_keypoints(image)      : cv2.ORB_create() -> (kp, desc)
    match_features(desc1, desc2) : cv2.BFMatcher (Brute-Force Hamming) -> good matches
    estimate_homography(...)     : cv2.findHomography(..., cv2.RANSAC) -> M, mask
    warp_images(image, M, shape) : cv2.warpPerspective -> registered image

Works on any image pair (placeholder or real downloads); produces a 3x3
homography matrix. Grayscale images are used for feature detection; the warped
output preserves the input channel count.
"""

import cv2
import numpy as np

ORB_MAX_FEATURES = 1000
RANSAC_THRESHOLD = 3.0          # pixels
LOWE_RATIO = 0.75               # good-match-to-second-best ratio (Lowe 2004)


def detect_keypoints(image):
    """Detect ORB keypoints and descriptors for a grayscale image.

    Returns (keypoints, descriptors). The image is converted to uint8
    grayscale internally so the function accepts colour or grey input.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 \
        else image.astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
    return orb.detectAndCompute(gray, None)


def match_features(desc1, desc2):
    """Brute-Force Hamming matcher with a Lowe ratio test.

    Returns the list of good cv2.DMatch objects (sorted by distance).
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)
    return sorted(good, key=lambda d: d.distance)


def estimate_homography(kp1, kp2, matches):
    """Estimate the 3x3 homography mapping image 1 -> image 2.

    Uses cv2.findHomography with RANSAC. Returns (M, mask) with M = None if
    the match set is too small to solve.
    """
    if len(matches) < 4:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    return M, mask


def warp_images(image, M, shape):
    """Apply homography M to image, returning an image of size `shape` (w, h)."""
    if M is None:
        return image
    h, w = shape
    return cv2.warpPerspective(image, M, (w, h))


def register_images(image_ref, image_moving):
    """Full registration chain: aligns image_moving onto image_ref.

    Returns (registered, H, inlier_frac) where H is the 3x3 homography that
    maps the *moving* image into the reference frame (used for the warp), and
    inlier_frac the RANSAC inlier ratio (0.0 if no homography was found).
    """
    kp1, desc1 = detect_keypoints(image_ref)
    kp2, desc2 = detect_keypoints(image_moving)
    matches = match_features(desc1, desc2)
    M, mask = estimate_homography(kp1, kp2, matches)  # M: ref -> moving
    if M is None or mask is None:
        return image_moving, M, 0.0
    H = np.linalg.inv(M)                              # invert: moving -> ref
    shape = image_ref.shape[:2] if image_ref.ndim == 2 else image_ref.shape[:2]
    registered = warp_images(image_moving, H, shape)
    inlier_frac = float(mask.sum() / max(len(matches), 1))
    return registered, H, inlier_frac


def _synthetic_pair(seed=0, size=400):
    """Build a textured reference image and a warped 'moving' copy."""
    rng = np.random.default_rng(seed)
    ref = np.zeros((size, size), np.uint8)
    for _ in range(120):
        cx, cy = rng.integers(15, size - 15, size=2)
        r = int(rng.integers(2, 10))
        cv2.circle(ref, (int(cx), int(cy)), r, int(rng.integers(40, 255)), -1)
    M_true = np.array([[1.0, 0.12, 18.0], [-0.09, 1.0, -12.0], [0.0, 0.0, 1.0]])
    moving = cv2.warpPerspective(ref, M_true, (size, size))
    return ref, moving, M_true


if __name__ == "__main__":
    ref_img, moving_img, M_true = _synthetic_pair()
    registered, H, inliers = register_images(ref_img, moving_img)

    print(f"keypoints matched: {inliers:.2%} inliers")
    print(f"recovered homography H =\n{H}")
    if H is not None:
        err = np.abs(np.linalg.inv(H) - M_true)
        assert err[0:2, 0:2].max() < 0.02, "linear terms drift"
        assert err[0:2, 2].max() < 1.0, f"translation drift {err[0:2, 2]}"
        assert err[2].max() < 0.05, "projective term drift"
        interior = np.s_[30:-30, 30:-30]
        diff_after = np.mean(np.abs(registered[interior].astype(np.int16)
                                    - ref_img[interior].astype(np.int16)))
        print(f"mean abs diff over interior (registered vs reference): "
              f"{diff_after:.2f} / 255")
        assert diff_after < 8.0, "registration did not align the moving image"
        print("PASS: pipeline ran without errors and produced a homography")