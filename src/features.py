"""Cheap, duration-agnostic video features.

Reads at most ~64 low-res frames per video regardless of length: WINDOWS short
windows of SECS consecutive 1-second-spaced samples, each with a +2-frame
partner used for optical flow. Cost is dominated by seeking/decoding, not math.
"""
import cv2
import numpy as np

WINDOWS = 4      # windows spread evenly across the video
SECS = 8         # 1 s-spaced samples per window
SIZE = 96        # analysis resolution (square, grayscale/colour)
FACE_SIZE = 160

FEATURE_NAMES = [
    "cut_rate", "hist_diff_med", "frame_diff_med",
    "flow_mag", "cam_motion", "flow_residual", "flow_coherence",
    "static_frac", "edge_density", "sat_mean", "color_entropy",
    "face_frac", "face_area", "brightness_std",
]

_FACE = None


def _face_detector():
    """Haar face detector, or None if this OpenCV build lacks it (features then 0)."""
    global _FACE
    if _FACE is None:
        try:
            _FACE = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if _FACE.empty():
                _FACE = False
        except AttributeError:
            _FACE = False
    return _FACE or None


def _read(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, f = cap.read()
    return f if ok else None


def _sample_indices(n_frames, fps):
    step = max(1, int(round(fps)))
    span = SECS * step
    if n_frames <= span + 3:
        starts = [0]
    else:
        starts = np.linspace(0, n_frames - span - 3, WINDOWS).astype(int)
        starts = sorted(set(starts.tolist()))
    return [[s + k * step for k in range(SECS) if s + k * step + 2 < n_frames]
            for s in starts]


def extract(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    wins = _sample_indices(n, fps)

    hist_d, frame_d, mags, cams, resid, cohs = [], [], [], [], [], []
    statics, edges, sats, ents, brights = [], [], [], [], []
    faces, face_area = [], []
    face_budget = 6

    for idxs in wins:
        grays, hists = [], []
        for i in idxs:
            a, b = _read(cap, i), _read(cap, i + 2)
            if a is None or b is None:
                continue
            a = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
            # optical flow on the close pair
            fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            mag = np.linalg.norm(fl, axis=2)
            med = np.median(fl.reshape(-1, 2), axis=0)
            mean_vec = fl.reshape(-1, 2).mean(0)
            mags.append(mag.mean())
            cams.append(np.linalg.norm(med))
            resid.append(np.linalg.norm(fl - med, axis=2).mean())
            cohs.append(np.linalg.norm(mean_vec) / (mag.mean() + 1e-3))
            # appearance stats
            edges.append((cv2.Canny(ga, 80, 160) > 0).mean())
            hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
            sats.append(hsv[..., 1].mean() / 255.0)
            q = (a >> 4).reshape(-1, 3).astype(np.int32)
            code = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
            p = np.bincount(code, minlength=4096) / len(code)
            p = p[p > 0]
            ents.append(float(-(p * np.log2(p)).sum()) / 12.0)
            brights.append(ga.mean() / 255.0)
            h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            hists.append(cv2.normalize(h, h).flatten())
            grays.append(ga.astype(np.float32))
            if face_budget > 0 and _face_detector() is not None:
                gf = cv2.cvtColor(cv2.resize(_read(cap, i), (FACE_SIZE, FACE_SIZE)),
                                  cv2.COLOR_BGR2GRAY)
                fs = _face_detector().detectMultiScale(gf, 1.15, 4, minSize=(20, 20))
                faces.append(1.0 if len(fs) else 0.0)
                face_area.append(max([w * h_ for (_, _, w, h_) in fs], default=0)
                                 / FACE_SIZE ** 2)
                face_budget -= 1
        for x, y in zip(hists[:-1], hists[1:]):
            hist_d.append(float(cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA)))
        for x, y in zip(grays[:-1], grays[1:]):
            frame_d.append(float(np.abs(x - y).mean()) / 255.0)
        if len(grays) >= 3:
            statics.append(float((np.std(np.stack(grays), axis=0) < 4.0).mean()))
    cap.release()
    if not mags:
        raise IOError(f"no decodable frames in {path}")

    g = lambda v, d=0.0: float(np.mean(v)) if len(v) else d
    m = lambda v, d=0.0: float(np.median(v)) if len(v) else d
    cut_rate = g(np.array(hist_d) > 0.5) if hist_d else 0.0
    return np.array([
        cut_rate, m(hist_d), m(frame_d),
        g(mags), g(cams), g(resid), g(cohs),
        g(statics), g(edges), g(sats), g(ents),
        g(faces), g(face_area), float(np.std(brights)),
    ], dtype=np.float32)
