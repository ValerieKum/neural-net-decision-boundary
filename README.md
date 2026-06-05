# Neural Net Learning a Decision Boundary

A small neural network learns a decision boundary from scratch, the boundary morphing as it trains.

Part of my portfolio of small, from-scratch visualisations of computer-science ideas. Built on numpy and matplotlib, so every moving part is visible.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python nn_decision_boundary.py                  # live animated window
python nn_decision_boundary.py --save out.gif   # export a looping GIF
python nn_decision_boundary.py --save out.mp4   # smaller file, best for the web (needs ffmpeg)
```
