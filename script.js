const STEPS = ["input", "learn", "output"];

const state = {
  algo: "linear",
  stepIndex: 0,
};

const refs = {
  algoButtons: [...document.querySelectorAll(".algo-btn")],
  stepIndicators: [...document.querySelectorAll(".step-indicator")],
  prevButton: document.getElementById("prev-step"),
  nextButton: document.getElementById("next-step"),
  progressBar: document.getElementById("progress-bar"),
  stageTitle: document.getElementById("stage-title"),
  stageCopy: document.getElementById("stage-copy"),
  watchCopy: document.getElementById("watch-copy"),
  linearPanel: document.getElementById("linear-panel"),
  deepPanel: document.getElementById("deep-panel"),
  linearInputControl: document.getElementById("linear-input-control"),
  deepSampleControl: document.getElementById("deep-sample-control"),
};

const stageCopyMap = {
  linear: {
    input: {
      title: "Input: collect examples",
      copy:
        "We start with example points. Each dot is one known data pair (input x, real y). The machine does not know the rule yet.",
      watch:
        "Notice the dots are scattered, not perfectly on one line. Real-life data is noisy like this.",
    },
    learn: {
      title: "Learn: find the best line",
      copy:
        "The model keeps adjusting one line so it sits as close to the cloud of dots as possible.",
      watch:
        "Watch the line tilt and slide until it matches the trend in the dots.",
    },
    output: {
      title: "Output: make a prediction",
      copy:
        "Now the learned line can predict new values. Move the slider to pick a fresh input, and the model gives an estimated output.",
      watch:
        "The highlighted point is a new prediction, not a training dot.",
    },
  },
  deep: {
    input: {
      title: "Input: describe features",
      copy:
        "This demo uses one sample slider. It sets feature values that enter the first layer of a neural network.",
      watch:
        "For now, keep the sample fixed and click Next to watch the network practice.",
    },
    learn: {
      title: "Learn: improve the network",
      copy:
        "The network practices with many examples and updates its connections little by little.",
      watch:
        "Just watch this step. The connecting lines shift as the model learns.",
    },
    output: {
      title: "Output: class probabilities",
      copy:
        "Now move the sample slider. The model returns probabilities for each class.",
      watch:
        "Try different sample values and watch which class bar becomes highest.",
    },
  },
};

const linear = {
  svg: document.getElementById("linear-svg"),
  slider: document.getElementById("linear-x-slider"),
  xReadout: document.getElementById("linear-x-readout"),
  stats: document.getElementById("linear-stats"),
  regenButton: document.getElementById("regen-data"),
  timer: null,
  data: [],
  m: 0.2,
  b: 8,
  targetM: 1,
  targetB: 5,
  round: 0,
  maxRounds: 96,
  initialError: null,
  currentError: null,
  margin: { top: 18, right: 22, bottom: 42, left: 58 },
  size: { w: 700, h: 360 },
};

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function linearGenerateData() {
  const trueM = rand(0.65, 1.45);
  const trueB = rand(2.0, 6.6);

  const points = [];
  for (let i = 0; i < 26; i += 1) {
    const x = rand(0.4, 9.6);
    const noise = rand(-1.2, 1.2);
    const y = clamp(trueM * x + trueB + noise, 0.4, 19.2);
    points.push({ x, y });
  }

  linear.data = points;
  linearResetTraining();
}

function linearResetTraining() {
  const bestFit = calcBestFitLine(linear.data);
  linear.targetM = bestFit.m;
  linear.targetB = bestFit.b;
  linear.m = rand(-0.45, 0.45);
  linear.b = rand(7.5, 13);
  linear.round = 0;
  linear.initialError = linearError(linear.m, linear.b);
  linear.currentError = linear.initialError;
  stopLinearTraining();
  renderLinear();
}

function linearPredict(x) {
  return linear.m * x + linear.b;
}

function linearError(m, b) {
  let sum = 0;
  for (const p of linear.data) {
    const err = Math.abs(m * p.x + b - p.y);
    sum += err;
  }
  return sum / linear.data.length;
}

function calcBestFitLine(points) {
  const n = points.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;

  for (const p of points) {
    sumX += p.x;
    sumY += p.y;
    sumXY += p.x * p.y;
    sumXX += p.x * p.x;
  }

  const denominator = n * sumXX - sumX * sumX;
  if (Math.abs(denominator) < 1e-9) {
    return { m: 0, b: sumY / n };
  }

  const m = (n * sumXY - sumX * sumY) / denominator;
  const b = (sumY - m * sumX) / n;
  return { m, b };
}

function linearTrainRound() {
  const blend = 0.115;
  linear.m += (linear.targetM - linear.m) * blend;
  linear.b += (linear.targetB - linear.b) * blend;
  linear.round += 1;
  linear.currentError = linearError(linear.m, linear.b);
}

function scaleX(x) {
  const { w } = linear.size;
  const { left, right } = linear.margin;
  return left + (x / 10) * (w - left - right);
}

function scaleY(y) {
  const { h } = linear.size;
  const { top, bottom } = linear.margin;
  return h - bottom - (y / 20) * (h - top - bottom);
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
  return el;
}

function drawLinearAxes(svg) {
  const g = svgEl("g", { opacity: "0.95" });
  const { w, h } = linear.size;
  const m = linear.margin;

  const axisColor = "#8fa2ba";
  g.appendChild(svgEl("line", { x1: m.left, y1: h - m.bottom, x2: w - m.right, y2: h - m.bottom, stroke: axisColor, "stroke-width": 1.4 }));
  g.appendChild(svgEl("line", { x1: m.left, y1: m.top, x2: m.left, y2: h - m.bottom, stroke: axisColor, "stroke-width": 1.4 }));

  for (let x = 0; x <= 10; x += 2) {
    const px = scaleX(x);
    g.appendChild(svgEl("line", { x1: px, y1: h - m.bottom, x2: px, y2: h - m.bottom + 7, stroke: axisColor, "stroke-width": 1 }));
    const label = svgEl("text", {
      x: px,
      y: h - m.bottom + 21,
      fill: "#5a6b86",
      "font-size": 11,
      "text-anchor": "middle",
    });
    label.textContent = String(x);
    g.appendChild(label);
  }

  for (let y = 0; y <= 20; y += 5) {
    const py = scaleY(y);
    g.appendChild(svgEl("line", { x1: m.left - 7, y1: py, x2: m.left, y2: py, stroke: axisColor, "stroke-width": 1 }));
    const label = svgEl("text", {
      x: m.left - 12,
      y: py + 4,
      fill: "#5a6b86",
      "font-size": 11,
      "text-anchor": "end",
    });
    label.textContent = String(y);
    g.appendChild(label);
  }

  const xCaption = svgEl("text", {
    x: w - m.right,
    y: h - 10,
    fill: "#52617f",
    "font-size": 11,
    "text-anchor": "end",
  });
  xCaption.textContent = "Input value (x)";
  g.appendChild(xCaption);

  const yCaption = svgEl("text", {
    x: 18,
    y: m.top - 2,
    fill: "#52617f",
    "font-size": 11,
  });
  yCaption.textContent = "Real/Predicted output (y)";
  g.appendChild(yCaption);

  svg.appendChild(g);
}

function renderLinear() {
  const svg = linear.svg;
  svg.innerHTML = "";
  drawLinearAxes(svg);

  const pointsGroup = svgEl("g");
  for (const p of linear.data) {
    pointsGroup.appendChild(
      svgEl("circle", {
        cx: scaleX(p.x),
        cy: scaleY(p.y),
        r: 4.5,
        fill: "#c0575f",
        opacity: 0.82,
      }),
    );
  }
  svg.appendChild(pointsGroup);

  const step = STEPS[state.stepIndex];
  if (step !== "input") {
    const y1 = clamp(linearPredict(0), 0, 20);
    const y2 = clamp(linearPredict(10), 0, 20);

    const line = svgEl("line", {
      x1: scaleX(0),
      y1: scaleY(y1),
      x2: scaleX(10),
      y2: scaleY(y2),
      stroke: "#0f8f8f",
      "stroke-width": 3.1,
      "stroke-linecap": "round",
    });
    svg.appendChild(line);
  }

  if (step === "output") {
    const x = Number(linear.slider.value);
    const y = clamp(linearPredict(x), 0, 20);

    svg.appendChild(
      svgEl("line", {
        x1: scaleX(x),
        y1: scaleY(0),
        x2: scaleX(x),
        y2: scaleY(y),
        stroke: "#d67436",
        "stroke-width": 2,
        "stroke-dasharray": "6 6",
      }),
    );

    svg.appendChild(
      svgEl("circle", {
        cx: scaleX(x),
        cy: scaleY(y),
        r: 6,
        fill: "#d67436",
      }),
    );

    const label = svgEl("text", {
      x: scaleX(x) + 9,
      y: scaleY(y) - 10,
      fill: "#94491f",
      "font-size": 12,
      "font-weight": 600,
    });
    label.textContent = `Prediction y=${y.toFixed(2)}`;
    svg.appendChild(label);
  }

  linear.xReadout.textContent = `x = ${Number(linear.slider.value).toFixed(1)}`;
  if (step === "input") {
    linear.stats.textContent = "These dots are known examples. Click Learn to let the line move into place.";
  } else if (step === "learn") {
    const fit = linear.initialError
      ? clamp(100 - (linear.currentError / linear.initialError) * 100, 0, 100)
      : 0;
    linear.stats.textContent = `Line fit progress: ${fit.toFixed(0)}%`;
  } else {
    const pred = linearPredict(Number(linear.slider.value));
    linear.stats.textContent = `For this input, the model predicts about ${pred.toFixed(1)}.`;
  }
}

function startLinearTraining() {
  stopLinearTraining();
  linear.timer = setInterval(() => {
    if (state.algo !== "linear" || STEPS[state.stepIndex] !== "learn") {
      stopLinearTraining();
      return;
    }

    if (linear.round >= linear.maxRounds || linear.currentError < 0.75) {
      stopLinearTraining();
      renderLinear();
      return;
    }

    for (let i = 0; i < 2 && linear.round < linear.maxRounds; i += 1) {
      linearTrainRound();
    }

    renderLinear();
  }, 40);
}

function stopLinearTraining() {
  if (linear.timer) {
    clearInterval(linear.timer);
    linear.timer = null;
  }
}

const deep = {
  svg: document.getElementById("network-svg"),
  profileSlider: document.getElementById("deep-profile"),
  profileReadout: document.getElementById("deep-profile-readout"),
  statusText: document.getElementById("deep-status"),
  guideText: document.getElementById("deep-guide"),
  barsWrap: document.getElementById("prediction-bars"),
  timer: null,
  round: 0,
  maxRounds: 180,
  lr: 0.08,
  inSize: 3,
  hiddenSize: 5,
  outSize: 3,
  w1: [],
  b1: [],
  w2: [],
  b2: [],
};

const classNames = ["Cat", "Dog", "Rabbit"];

function makeToyDataset() {
  const core = [
    { x: [0.82, 0.62, 0.55], label: 0 },
    { x: [0.75, 0.7, 0.48], label: 0 },
    { x: [0.66, 0.58, 0.64], label: 0 },
    { x: [0.35, 0.55, 0.84], label: 1 },
    { x: [0.28, 0.45, 0.74], label: 1 },
    { x: [0.42, 0.69, 0.88], label: 1 },
    { x: [0.9, 0.26, 0.35], label: 2 },
    { x: [0.8, 0.34, 0.28], label: 2 },
    { x: [0.93, 0.22, 0.4], label: 2 },
  ];

  const expanded = [];
  for (const item of core) {
    expanded.push(item);
    for (let i = 0; i < 2; i += 1) {
      expanded.push({
        x: item.x.map((v) => clamp(v + rand(-0.08, 0.08), 0, 1)),
        label: item.label,
      });
    }
  }
  return expanded;
}

const toyDataset = makeToyDataset();

function randomMatrix(rows, cols, range = 0.8) {
  const m = [];
  for (let r = 0; r < rows; r += 1) {
    const row = [];
    for (let c = 0; c < cols; c += 1) {
      row.push(rand(-range, range));
    }
    m.push(row);
  }
  return m;
}

function randomVector(n, range = 0.25) {
  return Array.from({ length: n }, () => rand(-range, range));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function deepInit() {
  deep.w1 = randomMatrix(deep.hiddenSize, deep.inSize);
  deep.b1 = randomVector(deep.hiddenSize);
  deep.w2 = randomMatrix(deep.outSize, deep.hiddenSize);
  deep.b2 = randomVector(deep.outSize);
  deep.round = 0;
  deep.profileSlider.value = "30";
  deep.statusText.textContent = "Status: Ready to practice";
}

function forwardPass(input) {
  const z1 = Array.from({ length: deep.hiddenSize }, (_, i) => {
    let sum = deep.b1[i];
    for (let j = 0; j < deep.inSize; j += 1) {
      sum += deep.w1[i][j] * input[j];
    }
    return sum;
  });

  const a1 = z1.map(sigmoid);

  const z2 = Array.from({ length: deep.outSize }, (_, k) => {
    let sum = deep.b2[k];
    for (let i = 0; i < deep.hiddenSize; i += 1) {
      sum += deep.w2[k][i] * a1[i];
    }
    return sum;
  });

  const probs = softmax(z2);
  return { z1, a1, z2, probs };
}

function oneHot(label) {
  return Array.from({ length: deep.outSize }, (_, i) => (i === label ? 1 : 0));
}

function trainOneEpoch() {
  for (const sample of toyDataset) {
    const y = oneHot(sample.label);
    const { a1, probs } = forwardPass(sample.x);

    const dz2 = probs.map((p, k) => p - y[k]);

    const dw2 = Array.from({ length: deep.outSize }, () => Array(deep.hiddenSize).fill(0));
    for (let k = 0; k < deep.outSize; k += 1) {
      for (let i = 0; i < deep.hiddenSize; i += 1) {
        dw2[k][i] = dz2[k] * a1[i];
      }
    }

    const dz1 = Array(deep.hiddenSize).fill(0);
    for (let i = 0; i < deep.hiddenSize; i += 1) {
      let sum = 0;
      for (let k = 0; k < deep.outSize; k += 1) {
        sum += deep.w2[k][i] * dz2[k];
      }
      dz1[i] = sum * a1[i] * (1 - a1[i]);
    }

    const dw1 = Array.from({ length: deep.hiddenSize }, () => Array(deep.inSize).fill(0));
    for (let i = 0; i < deep.hiddenSize; i += 1) {
      for (let j = 0; j < deep.inSize; j += 1) {
        dw1[i][j] = dz1[i] * sample.x[j];
      }
    }

    for (let k = 0; k < deep.outSize; k += 1) {
      for (let i = 0; i < deep.hiddenSize; i += 1) {
        deep.w2[k][i] -= deep.lr * dw2[k][i];
      }
      deep.b2[k] -= deep.lr * dz2[k];
    }

    for (let i = 0; i < deep.hiddenSize; i += 1) {
      for (let j = 0; j < deep.inSize; j += 1) {
        deep.w1[i][j] -= deep.lr * dw1[i][j];
      }
      deep.b1[i] -= deep.lr * dz1[i];
    }
  }

  deep.round += 1;
}

function getFeatureInput() {
  const anchors = [
    [0.84, 0.6, 0.5],
    [0.35, 0.56, 0.86],
    [0.9, 0.26, 0.34],
    [0.84, 0.6, 0.5],
  ];
  const t = Number(deep.profileSlider.value) / 100;
  const scaled = t * 3;
  const idx = Math.min(2, Math.floor(scaled));
  const local = idx === 2 ? Math.min(1, scaled - 2) : scaled - idx;
  const a = anchors[idx];
  const b = anchors[idx + 1];
  return a.map((v, i) => v + (b[i] - v) * local);
}

function weightColor(w) {
  if (w >= 0) {
    return "rgba(15, 143, 143, 0.62)";
  }
  return "rgba(214, 116, 54, 0.62)";
}

function drawEdge(svg, x1, y1, x2, y2, w) {
  const thickness = 0.9 + Math.min(4.1, Math.abs(w) * 3.3);
  svg.appendChild(
    svgEl("line", {
      x1,
      y1,
      x2,
      y2,
      stroke: weightColor(w),
      "stroke-width": thickness,
      "stroke-linecap": "round",
    }),
  );
}

function drawNode(svg, x, y, activation, label) {
  const intensity = Math.round(230 - activation * 130);
  svg.appendChild(
    svgEl("circle", {
      cx: x,
      cy: y,
      r: 18,
      fill: `rgb(${intensity}, ${245 - Math.round(activation * 52)}, 233)`,
      stroke: "#263b52",
      "stroke-width": 1.2,
    }),
  );

  if (label) {
    const text = svgEl("text", {
      x,
      y: y + 4,
      "font-size": 10,
      "text-anchor": "middle",
      fill: "#25354a",
      "font-weight": 600,
    });
    text.textContent = label;
    svg.appendChild(text);
  }
}

function drawLayerLabel(svg, x, text) {
  const t = svgEl("text", {
    x,
    y: 22,
    "font-size": 12,
    "text-anchor": "middle",
    fill: "#52617f",
    "font-weight": 600,
  });
  t.textContent = text;
  svg.appendChild(t);
}

function renderDeepNetwork() {
  const step = STEPS[state.stepIndex];
  const input = getFeatureInput();
  const { a1, probs } = forwardPass(input);
  const svg = deep.svg;
  svg.innerHTML = "";

  const xPos = { in: 100, hidden: 350, out: 600 };
  const inY = [72, 170, 268];
  const hiddenY = [44, 108, 172, 236, 300];
  const outY = [94, 172, 250];

  drawLayerLabel(svg, xPos.in, "Input features");
  drawLayerLabel(svg, xPos.hidden, "Hidden layer");
  drawLayerLabel(svg, xPos.out, "Output classes");

  for (let i = 0; i < deep.hiddenSize; i += 1) {
    for (let j = 0; j < deep.inSize; j += 1) {
      drawEdge(svg, xPos.in, inY[j], xPos.hidden, hiddenY[i], deep.w1[i][j]);
    }
  }

  for (let k = 0; k < deep.outSize; k += 1) {
    for (let i = 0; i < deep.hiddenSize; i += 1) {
      drawEdge(svg, xPos.hidden, hiddenY[i], xPos.out, outY[k], deep.w2[k][i]);
    }
  }

  const inputLabels = ["E", "F", "T"];
  for (let j = 0; j < deep.inSize; j += 1) {
    drawNode(svg, xPos.in, inY[j], input[j], inputLabels[j]);
  }

  for (let i = 0; i < deep.hiddenSize; i += 1) {
    drawNode(svg, xPos.hidden, hiddenY[i], a1[i]);
  }

  for (let k = 0; k < deep.outSize; k += 1) {
    drawNode(svg, xPos.out, outY[k], probs[k], classNames[k][0]);
  }

  renderPredictionBars(probs);
  deep.profileReadout.textContent = `Sample: ${deep.profileSlider.value}`;

  if (step === "input") {
    deep.guideText.textContent = "Next action: click Next to start the learning animation.";
  } else if (step === "learn") {
    deep.guideText.textContent = "Next action: watch the network practice, then click Next.";
  } else {
    deep.guideText.textContent = "Next action: move the sample slider to test new inputs.";
  }

  if (deep.round === 0 && step !== "learn") {
    deep.statusText.textContent = "Status: Ready to practice";
  } else if (deep.round < deep.maxRounds) {
    deep.statusText.textContent = "Status: Practicing with examples";
  } else {
    deep.statusText.textContent = "Status: Ready to predict";
  }
}

function renderPredictionBars(probs) {
  deep.barsWrap.innerHTML = "";
  probs.forEach((p, idx) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${classNames[idx]}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span>${(p * 100).toFixed(1)}%</span>
    `;
    deep.barsWrap.appendChild(row);
  });
}

function stopDeepTraining() {
  if (deep.timer) {
    clearInterval(deep.timer);
    deep.timer = null;
  }
}

function startDeepTraining(autoToMax = false) {
  stopDeepTraining();
  deep.timer = setInterval(() => {
    if (state.algo !== "deep") {
      stopDeepTraining();
      return;
    }

    const step = STEPS[state.stepIndex];
    if (step !== "learn" && autoToMax) {
      stopDeepTraining();
      return;
    }

    if (deep.round >= deep.maxRounds) {
      stopDeepTraining();
      renderDeepNetwork();
      return;
    }

    for (let i = 0; i < 2 && deep.round < deep.maxRounds; i += 1) {
      trainOneEpoch();
    }
    renderDeepNetwork();
  }, 50);
}

function setStep(index) {
  state.stepIndex = clamp(index, 0, STEPS.length - 1);
  updateUiState();
}

function setAlgorithm(algo) {
  if (state.algo === algo) {
    return;
  }

  state.algo = algo;
  state.stepIndex = 0;

  if (algo === "linear") {
    linearResetTraining();
  } else {
    deepInit();
  }

  updateUiState();
}

function updateUiState() {
  const step = STEPS[state.stepIndex];

  refs.algoButtons.forEach((btn) => {
    const isActive = btn.dataset.algo === state.algo;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-selected", String(isActive));
  });

  refs.stepIndicators.forEach((el, idx) => {
    el.classList.toggle("active", idx === state.stepIndex);
  });

  refs.linearPanel.classList.toggle("hidden", state.algo !== "linear");
  refs.deepPanel.classList.toggle("hidden", state.algo !== "deep");
  refs.linearInputControl.classList.toggle(
    "hidden",
    !(state.algo === "linear" && step === "output"),
  );
  refs.deepSampleControl.classList.toggle(
    "hidden",
    !(state.algo === "deep" && step === "output"),
  );
  deep.profileSlider.disabled = !(state.algo === "deep" && step === "output");

  refs.prevButton.disabled = state.stepIndex === 0;
  refs.nextButton.disabled = state.stepIndex === STEPS.length - 1;

  refs.progressBar.style.width = `${((state.stepIndex + 1) / STEPS.length) * 100}%`;

  const copy = stageCopyMap[state.algo][step];
  refs.stageTitle.textContent = copy.title;
  refs.stageCopy.textContent = copy.copy;
  refs.watchCopy.textContent = copy.watch;

  if (state.algo === "linear") {
    stopDeepTraining();
    if (step === "learn") {
      startLinearTraining();
    } else {
      stopLinearTraining();
      renderLinear();
    }
  } else {
    stopLinearTraining();
    if (step === "learn") {
      startDeepTraining(true);
    } else {
      stopDeepTraining();

      if (step === "output" && deep.round < 40) {
        for (let i = deep.round; i < 40; i += 1) {
          trainOneEpoch();
        }
      }
      renderDeepNetwork();
    }
  }
}

function bindEvents() {
  refs.algoButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      setAlgorithm(btn.dataset.algo);
    });
  });

  refs.prevButton.addEventListener("click", () => setStep(state.stepIndex - 1));
  refs.nextButton.addEventListener("click", () => setStep(state.stepIndex + 1));

  linear.slider.addEventListener("input", renderLinear);
  linear.regenButton.addEventListener("click", linearGenerateData);

  deep.profileSlider.addEventListener("input", renderDeepNetwork);

  window.addEventListener("beforeunload", () => {
    stopLinearTraining();
    stopDeepTraining();
  });
}

function init() {
  linearGenerateData();
  deepInit();
  bindEvents();
  updateUiState();
}

init();
