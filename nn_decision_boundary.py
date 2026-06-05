"""A tiny neural network learning to separate two tangled spirals.

Built from scratch in numpy — no TensorFlow, no PyTorch — so you can see every
moving part: a 2-layer net (tanh hidden layer + sigmoid output) trained by
backprop. The colored background is the network's *decision boundary*. It
starts as mush and warps, frame by frame, until it wraps cleanly around each
spiral arm. That warping IS the learning.

Run:
    python learning/nn_decision_boundary.py
    python learning/nn_decision_boundary.py --save nn.gif
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.viz import (PALETTE, card_figure, caption, save_animation,
                        use_headless_if_saving)

# ---- tunables -------------------------------------------------------------
HIDDEN = 32
LR = 1.0
STEPS_PER_FRAME = 14
TOTAL_FRAMES = 140
TURNS = 1.5            # spiral tightness — 1.5 is learnable by one hidden layer
GRID_RES = 90          # decision-boundary mesh resolution (keep small = micro)


def make_spirals(n=200, turns=TURNS, noise=0.06):
    """Two interleaving spiral arms — a classic non-linear toy problem.

    Inputs are standardized (zero mean, unit variance) so training converges
    reliably regardless of the random spiral.
    """
    t = np.linspace(0, 1, n)
    X, y = [], []
    for cls in (0, 1):
        a = t * turns * 2 * np.pi + cls * np.pi
        xs = t * np.cos(a) + np.random.randn(n) * noise
        ys = t * np.sin(a) + np.random.randn(n) * noise
        X.append(np.c_[xs, ys])
        y += [cls] * n
    X = np.vstack(X)
    X = (X - X.mean(0)) / X.std(0)
    return X, np.array(y, dtype=np.float64).reshape(-1, 1)


def sigmoid(z):
    """Overflow-safe logistic (branch on sign so exp never blows up)."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class Net:
    """2-layer MLP: in(2) -> tanh(HIDDEN) -> sigmoid(1)."""

    def __init__(self, hidden):
        rng = np.random.randn
        # Xavier-ish init keeps activations well-scaled from the start
        self.W1, self.b1 = rng(2, hidden) * np.sqrt(1.0), np.zeros((1, hidden))
        self.W2, self.b2 = rng(hidden, 1) * np.sqrt(1.0 / hidden), np.zeros((1, 1))

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def step(self, y, lr):
        m = len(y)
        d2 = (self.out - y) / m
        dW2 = self.a1.T @ d2
        db2 = d2.sum(0, keepdims=True)
        d1 = (d2 @ self.W2.T) * (1 - self.a1 ** 2)
        dW1 = self.X.T @ d1
        db1 = d1.sum(0, keepdims=True)
        # guard against blow-ups: skip non-finite grads, then clip the norm
        grads = [dW1, db1, dW2, db2]
        if not all(np.isfinite(g).all() for g in grads):
            return float("nan")
        norm = np.sqrt(sum((g ** 2).sum() for g in grads))
        if norm > 10.0:
            dW1, db1, dW2, db2 = (g * (10.0 / norm) for g in grads)
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        return float(np.mean(-(y * np.log(self.out + 1e-9) +
                               (1 - y) * np.log(1 - self.out + 1e-9))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="OUT.gif")
    args = ap.parse_args()
    use_headless_if_saving(args.save)
    # transient overflows on a bad random init are caught by the step guard;
    # silence the benign float warnings they emit
    np.seterr(over="ignore", invalid="ignore", divide="ignore")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap

    X, y = make_spirals()
    net = Net(HIDDEN)

    lim = float(np.abs(X).max()) * 1.12
    gx, gy = np.meshgrid(np.linspace(-lim, lim, GRID_RES),
                         np.linspace(-lim, lim, GRID_RES))
    mesh = np.c_[gx.ravel(), gy.ravel()]

    cmap = LinearSegmentedColormap.from_list(
        "spiral", [PALETTE["cool"], PALETTE["bg"], PALETTE["pink"]])

    fig, ax = card_figure()
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    field = ax.imshow(np.zeros((GRID_RES, GRID_RES)), extent=(-lim, lim, -lim, lim),
                      origin="lower", cmap=cmap, vmin=0, vmax=1, alpha=0.85)
    cls0 = y.ravel() == 0
    ax.scatter(X[cls0, 0], X[cls0, 1], s=14, color=PALETTE["cyan"],
               edgecolors=PALETTE["bg"], linewidths=0.5, zorder=3)
    ax.scatter(X[~cls0, 0], X[~cls0, 1], s=14, color=PALETTE["hot"],
               edgecolors=PALETTE["bg"], linewidths=0.5, zorder=3)
    txt = caption(ax, "")

    def render(frame):
        loss = 0.0
        for _ in range(STEPS_PER_FRAME):
            net.forward(X)
            loss = net.step(y, LR)
        field.set_array(net.forward(mesh).reshape(GRID_RES, GRID_RES))
        txt.set_text(f"epoch {frame * STEPS_PER_FRAME:>4}   loss {loss:.3f}")
        return field, txt

    anim = FuncAnimation(fig, render, frames=TOTAL_FRAMES, interval=50,
                         blit=True)

    if args.save:
        save_animation(anim, args.save, fps=20)
    else:
        plt.show()


if __name__ == "__main__":
    main()
