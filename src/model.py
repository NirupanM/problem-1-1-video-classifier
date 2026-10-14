"""Tiny softmax regression in numpy (~70 weights) + class -> chunking strategy."""
import json
import numpy as np

CLASSES = ["static_talking", "edited_multishot", "egocentric_continuous",
           "screen_synthetic", "static_event"]

STRATEGIES = {
    "static_talking": "speech-turn / ASR-boundary chunks (~30-60 s), 1 keyframe per chunk",
    "edited_multishot": "shot-boundary chunks (one chunk per shot), keyframe at shot centre",
    "egocentric_continuous": "motion-adaptive chunks: cut on activity/scene change, dense overlapping windows (~5-10 s)",
    "screen_synthetic": "UI-state change chunks (frame-diff on non-HUD region) + OCR/event cues",
    "static_event": "event-triggered chunks: ignore idle, chunk around motion bursts with pre/post padding",
}
FALLBACK = "generic fixed-window chunks (10 s, 25% overlap)"


class Model:
    def __init__(self, W=None, b=None, mu=None, sd=None):
        self.W, self.b, self.mu, self.sd = W, b, mu, sd

    def proba(self, X):
        Z = ((np.atleast_2d(X) - self.mu) / self.sd) @ self.W + self.b
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def fit(self, X, y, l2=1e-2, lr=0.3, epochs=800):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        Y = np.eye(len(CLASSES))[y]
        self.W = np.zeros((X.shape[1], len(CLASSES)))
        self.b = np.zeros(len(CLASSES))
        for _ in range(epochs):
            G = (self.proba(X) - Y) / len(X)
            self.W -= lr * (Xs.T @ G + l2 * self.W)
            self.b -= lr * G.sum(0)
        return self

    def predict(self, x, min_conf=0.6):
        p = self.proba(x)[0]
        k = int(p.argmax())
        if p[k] < min_conf:  # open-set guard: unsure -> generic chunking
            return "unknown", float(p[k]), FALLBACK
        return CLASSES[k], float(p[k]), STRATEGIES[CLASSES[k]]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({k: getattr(self, k).tolist() for k in ("W", "b", "mu", "sd")}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(*(np.array(d[k]) for k in ("W", "b", "mu", "sd")))
