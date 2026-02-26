# Machine Learning Methods Visualizer

A split-page, beginner-friendly website that teaches machine learning with clear guided paths.

## Pages

- `index.html` (Home): choose a learning path from two big start cards.
- `linear.html`: Linear Regression storyboard (`Input -> Learn -> Output`).
- `ann.html`: Deep Learning storyboard (`Input -> Learn -> Output`) using a spam-filter narrative.

## Learning Flows

### Linear Regression page

1. Input: click **Load Example Points**
2. Learn: click **Fit Trend Line**
3. Output: move the `x` slider to test predictions

### Deep Learning page

1. Input: click **Load 5 Sample Messages**
2. Learn: click **Load 500 Examples** then **Learn Fast**
3. Output: move the **Message risk level** slider to predict `Spam` vs `Not Spam`

ANN learning path uses this narrative:

- Start from rough guesses on 5 examples
- Train quickly on 500 examples
- Use the trained predictor with confidence (probability)

## Run locally

Open `index.html` directly, or run a local server:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Files

- `index.html`: Home chooser page
- `linear.html`: Linear Regression page
- `ann.html`: Deep Learning page
- `styles.css`: shared styling across all pages
- `linear.js`: linear-only state machine and rendering
- `ann.js`: ANN-only state machine and rendering
