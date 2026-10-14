"""Procedural stand-in videos, one generator per class. For plumbing/tests only:
accuracy on these says nothing about real footage."""
import cv2
import numpy as np

W, H, FPS, SECONDS = 160, 120, 15, 36


def _texture(rng, scale=8):
    t = rng.integers(0, 255, (H // scale + 2, W // scale + 2, 3), dtype=np.uint8)
    return cv2.resize(t, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)


def _crop(big, x, y):
    x = int(np.clip(x, 0, big.shape[1] - W))
    y = int(np.clip(y, 0, big.shape[0] - H))
    return big[y:y + H, x:x + W].copy()


def static_talking(rng, n):
    bg = np.full((H, W, 3), rng.integers(60, 160, 3), np.uint8)
    cv2.rectangle(bg, (0, 0), (W, 30), tuple(int(v) for v in rng.integers(30, 200, 3)), -1)
    for i in range(n):
        f = bg.copy()
        cx, cy = W // 2 + int(2 * np.sin(i / 7)), H // 2 + int(1.5 * np.sin(i / 3))
        cv2.ellipse(f, (cx, cy), (28, 36), 0, 0, 360, (150, 170, 220), -1)
        cv2.circle(f, (cx - 9, cy - 8), 3, (30, 30, 30), -1)
        cv2.circle(f, (cx + 9, cy - 8), 3, (30, 30, 30), -1)
        cv2.ellipse(f, (cx, cy + 14), (8, 2 + int(2 * abs(np.sin(i / 2)))), 0, 0, 360, (40, 40, 90), -1)
        yield f


def edited_multishot(rng, n):
    shot_len = int(rng.integers(2, 4) * FPS)
    for i in range(n):
        if i % shot_len == 0:
            big = _texture(rng, int(rng.integers(6, 20)))
            vx, vy = rng.normal(0, 2), rng.normal(0, 2)
            x, y = W // 2, H // 2
        x += vx
        y += vy
        f = _crop(big, x, y)
        cv2.circle(f, (int(40 + (i % shot_len) * 3), 60), 10, (255, 255, 255), -1)
        yield f


def egocentric_continuous(rng, n):
    big = _texture(rng, int(rng.integers(6, 14)))
    x, y = W // 2, H // 2
    for i in range(n):
        x += 4 * np.sin(i / 9) + rng.normal(0, 2.5)
        y += 3 * np.cos(i / 11) + rng.normal(0, 2.5)
        x = np.clip(x, 0, W)
        y = np.clip(y, 0, H)
        yield _crop(big, x, y)


def screen_synthetic(rng, n):
    cols = [tuple(int(v) for v in rng.integers(0, 255, 3)) for _ in range(3)]
    bg = np.full((H, W, 3), cols[0], np.uint8)
    for i in range(0, W, 16):
        cv2.line(bg, (i, 0), (i, H), cols[1], 1)
    cv2.rectangle(bg, (0, H - 14), (W, H), (20, 20, 20), -1)
    cv2.rectangle(bg, (4, 4), (50, 12), (0, 200, 0), -1)
    cv2.putText(bg, "HP 100", (60, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
    px = 20.0
    for i in range(n):
        f = bg.copy()
        px = (px + 3) % W
        py = 40 + 25 * np.sin(i / 10)
        cv2.rectangle(f, (int(px), int(py)), (int(px) + 12, int(py) + 12), cols[2], -1)
        if i % 90 == 0:
            bg = np.roll(bg, 16, axis=1)
        yield f


def static_event(rng, n):
    bg = _texture(rng, 5)[:H, :W]
    bg = cv2.addWeighted(bg, 0.3, np.full_like(bg, 110), 0.7, 0)
    ev = int(rng.integers(n // 4, n // 2))
    for i in range(n):
        f = np.clip(bg.astype(np.int16) + rng.integers(-2, 3, bg.shape), 0, 255).astype(np.uint8)
        if ev <= i < ev + 60:
            cv2.ellipse(f, (10 + (i - ev) * 2, 80), (14, 7), 0, 0, 360, (30, 40, 60), -1)
        yield f


GENERATORS = {
    "static_talking": static_talking,
    "edited_multishot": edited_multishot,
    "egocentric_continuous": egocentric_continuous,
    "screen_synthetic": screen_synthetic,
    "static_event": static_event,
}


def write(cls, path, seed):
    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in GENERATORS[cls](rng, FPS * SECONDS):
        vw.write(f)
    vw.release()
