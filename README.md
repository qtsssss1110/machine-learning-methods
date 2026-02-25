# Machine Learning Methods Visualizer

An interactive, beginner-friendly website that visually explains how machine learning works through three steps:

1. Input
2. Learn
3. Output / Prediction

It includes two modes:

- Linear Regression: scattered data points and an animated best-fit line moving into place.
- Deep Learning (ANN): feature inputs flowing through a neural network with practice rounds and class probabilities.

## Run locally

Because this is a static site, you can open `index.html` directly in a browser.

For a local server (recommended):

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Files

- `index.html`: page structure and educational content
- `styles.css`: visual design, layout, and responsive behavior
- `script.js`: interactive controls, visual simulations, and step logic
