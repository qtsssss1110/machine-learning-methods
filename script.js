const STEPS = ["input", "learn", "output"];

const state = {
  algo: "linear",
  stepIndex: 0,
};

const refs = {
  algoButtons: [...document.querySelectorAll(".algo-btn")],
  stepButtons: [...document.querySelectorAll(".step-btn")],
  prevButton: document.getElementById("prev-step"),
  nextButton: document.getElementById("next-step"),
  progressBar: document.getElementById("progress-bar"),
  stageTitle: document.getElementById("stage-title"),
  stageCopy: document.getElementById("stage-copy"),
  watchCopy: document.getElementById("watch-copy"),
  linearPanel: document.getElementById("linear-panel"),
  deepPanel: document.getElementById("deep-panel"),
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
        "The model keeps adjusting a line to reduce mistakes. Each training epoch nudges slope and intercept so predictions get closer to real dots.",
      watch:
        "Watch the line tilt and move while loss drops. Lower loss means better fit.",
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
        "Here each input is a set of features (ear shape, fur, tail). These numbers enter the first layer of a neural network.",
      watch:
        "Move sliders to change the sample. Brighter input nodes mean stronger feature values.",
    },
    learn: {
      title: "Learn: tune many weights",
      copy:
        "A neural network learns by updating many connection weights over many epochs. It compares guesses to true labels and reduces error.",
      watch:
        "Edge color and thickness show weight direction and strength while loss trends down.",
    },
    output: {
      title: "Output: class probabilities",
      copy:
        "The final layer returns probabilities for each class. The biggest bar is the model's best prediction for this sample.",
      watch:
        "Try new feature values to see how probability bars shift.",
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
  trueM: 1,
  trueB: 3,
  m: 0.2,
  b: 8,
  epoch: 0,
  maxEpoch: 140,
  lr: 0.032,
  lastLoss: null,
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
  linear.trueM = rand(0.65, 1.45);
  linear.trueB = rand(2.0, 6.6);

  const points = [];
  for (let i = 0; i < 26; i += 1) {
    const x = rand(0.4, 9.6);
    const noise = rand(-1.7, 1.7);
    const y = clamp(linear.trueM * x + linear.trueB + noise, 0.4, 19.2);
    points.push({ x, y });
  }

  linear.data = points;
  linearResetTraining();
}

function linearResetTraining() {
  linear.m = rand(-0.5, 0.5);
  linear.b = rand(6.5, 12.5);
  linear.epoch = 0;
  linear.lastLoss = null;
  stopLinearTraining();
  renderLinear();
}

function linearPredict(x) {
  return linear.m * x + linear.b;
}

function linearLoss(m, b) {
  let sum = 0;
  for (const p of linear.data) {
    const err = m * p.x + b - p.y;
    sum += err * err;
  }
  return sum / linear.data.length;
}

function linearTrainEpoch() {
  let dm = 0;
  let db = 0;
  const n = linear.data.length;

  for (const p of linear.data) {
    const err = linear.m * p.x + linear.b - p.y;
    dm += (2 / n) * err * p.x;
    db += (2 / n) * err;
  }

  linear.m -= linear.lr * dm;
  linear.b -= linear.lr * db;
  linear.epoch += 1;
  linear.lastLoss = linearLoss(linear.m, linear.b);
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
    linear.stats.textContent = "Dataset loaded. Click Learn to train the model.";
  } else if (step === "learn") {
    const lossText = linear.lastLoss === null ? "--" : linear.lastLoss.toFixed(3);
    linear.stats.textContent = `Epoch ${linear.epoch}/${linear.maxEpoch} | Loss ${lossText} | y = ${linear.m.toFixed(2)}x + ${linear.b.toFixed(2)}`;
  } else {
    const pred = linearPredict(Number(linear.slider.value));
    linear.stats.textContent = `Learned line: y = ${linear.m.toFixed(2)}x + ${linear.b.toFixed(2)} | Current prediction: ${pred.toFixed(2)}`;
  }
}

function startLinearTraining() {
  stopLinearTraining();
  linear.timer = setInterval(() => {
    if (state.algo !== "linear" || STEPS[state.stepIndex] !== "learn") {
      stopLinearTraining();
      return;
    }

    if (linear.epoch >= linear.maxEpoch) {
      stopLinearTraining();
      renderLinear();
      return;
    }

    for (let i = 0; i < 2 && linear.epoch < linear.maxEpoch; i += 1) {
      linearTrainEpoch();
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
  sliders: {
    ears: document.getElementById("feat-ears"),
    fur: document.getElementById("feat-fur"),
    tail: document.getElementById("feat-tail"),
  },
  trainButton: document.getElementById("deep-train"),
  randomButton: document.getElementById("deep-randomize"),
  epochText: document.getElementById("deep-epoch"),
  lossText: document.getElementById("deep-loss"),
  barsWrap: document.getElementById("prediction-bars"),
  lossCanvas: document.getElementById("loss-canvas"),
  timer: null,
  epoch: 0,
  maxEpoch: 180,
  lr: 0.08,
  history: [],
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
  deep.epoch = 0;
  deep.history = [];
  deep.lossText.textContent = "Loss: --";
  deep.epochText.textContent = "Epoch: 0";
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
  let epochLoss = 0;

  for (const sample of toyDataset) {
    const y = oneHot(sample.label);
    const { a1, probs } = forwardPass(sample.x);

    let sampleLoss = 0;
    for (let k = 0; k < deep.outSize; k += 1) {
      sampleLoss += -y[k] * Math.log(probs[k] + 1e-9);
    }
    epochLoss += sampleLoss;

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

  const avgLoss = epochLoss / toyDataset.length;
  deep.epoch += 1;
  deep.history.push(avgLoss);
}

function getFeatureInput() {
  return [
    Number(deep.sliders.ears.value),
    Number(deep.sliders.fur.value),
    Number(deep.sliders.tail.value),
  ];
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
  deep.epochText.textContent = `Epoch: ${deep.epoch}`;
  const currentLoss = deep.history.at(-1);
  deep.lossText.textContent = `Loss: ${currentLoss ? currentLoss.toFixed(3) : "--"}`;
  drawLossChart();
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

function drawLossChart() {
  const canvas = deep.lossCanvas;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(18, 23, 36, 0.2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(28, 12);
  ctx.lineTo(28, height - 20);
  ctx.lineTo(width - 10, height - 20);
  ctx.stroke();

  ctx.fillStyle = "#55637f";
  ctx.font = "11px Sora";
  ctx.fillText("Loss", 8, 18);
  ctx.fillText("Epoch", width - 42, height - 6);

  if (deep.history.length < 2) {
    return;
  }

  const minLoss = Math.min(...deep.history);
  const maxLoss = Math.max(...deep.history);
  const span = Math.max(0.001, maxLoss - minLoss);

  ctx.strokeStyle = "#0f8f8f";
  ctx.lineWidth = 2;
  ctx.beginPath();

  deep.history.forEach((loss, i) => {
    const x = 28 + (i / (deep.history.length - 1)) * (width - 42);
    const y = 12 + ((maxLoss - loss) / span) * (height - 32);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
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

    if (deep.epoch >= deep.maxEpoch) {
      stopDeepTraining();
      renderDeepNetwork();
      return;
    }

    for (let i = 0; i < 2 && deep.epoch < deep.maxEpoch; i += 1) {
      trainOneEpoch();
    }
    renderDeepNetwork();
  }, 50);
}

function randomizeFeatureInput() {
  deep.sliders.ears.value = rand(0.15, 0.95).toFixed(2);
  deep.sliders.fur.value = rand(0.1, 0.9).toFixed(2);
  deep.sliders.tail.value = rand(0.12, 0.95).toFixed(2);
  renderDeepNetwork();
}

function setStep(index) {
  state.stepIndex = clamp(index, 0, STEPS.length - 1);
  updateUiState();
}

function setAlgorithm(algo) {
  state.algo = algo;
  updateUiState();
}

function updateUiState() {
  const step = STEPS[state.stepIndex];

  refs.algoButtons.forEach((btn) => {
    const isActive = btn.dataset.algo === state.algo;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-selected", String(isActive));
  });

  refs.stepButtons.forEach((btn, idx) => {
    const isActive = idx === state.stepIndex;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-selected", String(isActive));
  });

  refs.linearPanel.classList.toggle("hidden", state.algo !== "linear");
  refs.deepPanel.classList.toggle("hidden", state.algo !== "deep");

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

      if (step === "output" && deep.epoch < 25) {
        for (let i = deep.epoch; i < 25; i += 1) {
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

  refs.stepButtons.forEach((btn, idx) => {
    btn.addEventListener("click", () => setStep(idx));
  });

  refs.prevButton.addEventListener("click", () => setStep(state.stepIndex - 1));
  refs.nextButton.addEventListener("click", () => setStep(state.stepIndex + 1));

  linear.slider.addEventListener("input", renderLinear);
  linear.regenButton.addEventListener("click", linearGenerateData);

  Object.values(deep.sliders).forEach((slider) => {
    slider.addEventListener("input", renderDeepNetwork);
  });

  deep.randomButton.addEventListener("click", randomizeFeatureInput);

  deep.trainButton.addEventListener("click", () => {
    if (deep.epoch >= deep.maxEpoch) {
      deepInit();
      renderDeepNetwork();
    }
    startDeepTraining(false);
  });

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
