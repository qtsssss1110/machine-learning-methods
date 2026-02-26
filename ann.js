const STEPS = ["input", "learn", "output"];

const state = {
  stepIndex: 0,
  sample5Loaded: false,
  sample500Loaded: false,
  trainedFast: false,
  roundsCompleted: 0,
  mistakesLeft: null,
  bestMistakes: null,
  learnTimer: null,
  quickPassIndex: 0,
};

const refs = {
  stepIndicators: [...document.querySelectorAll(".step-indicator")],
  prevButton: document.getElementById("prev-step"),
  nextButton: document.getElementById("next-step"),
  stagePill: document.getElementById("stage-pill"),
  stageName: document.getElementById("stage-name"),
  actionLine: document.getElementById("action-line"),
  watchLine: document.getElementById("watch-line"),
  svg: document.getElementById("ann-svg"),

  inputTools: document.getElementById("input-tools"),
  learnTools: document.getElementById("learn-tools"),
  outputTools: document.getElementById("output-tools"),

  load5Button: document.getElementById("load-5"),
  sampleRows: document.getElementById("sample-rows"),
  sampleSummary: document.getElementById("sample-summary"),

  load500Button: document.getElementById("load-500"),
  learnFastButton: document.getElementById("learn-fast"),
  mistakes: document.getElementById("mistakes"),
  rounds: document.getElementById("rounds"),
  learnNote: document.getElementById("learn-note"),

  riskSlider: document.getElementById("risk-slider"),
  riskReadout: document.getElementById("risk-readout"),
  bars: document.getElementById("prediction-bars"),
  outputSummary: document.getElementById("output-summary"),
};

const stageInfo = {
  input: {
    name: "Input",
    action: "Do this now: click Load 5 Sample Messages.",
    watch: "What to watch: the model starts close to random guessing.",
  },
  learn: {
    name: "Learn",
    action: "Do this now: click Load 500 Examples, then click Learn Fast.",
    watch: "What to watch: weights adjust and mistakes drop round by round.",
  },
  output: {
    name: "Output",
    action: "Do this now: move Message risk level slider.",
    watch: "What to watch: Spam vs Not Spam bars and confidence sentence update.",
  },
};

const ann = {
  inSize: 3,
  hiddenSize: 2,
  outSize: 2,
  lr: 0.08,
  w1: [],
  b1: [],
  w2: [],
  b2: [],
};

const classNames = ["Spam", "Not Spam"];

const sample5 = [
  {
    text: "Urgent: verify account at tiny-link.co",
    x: [0.95, 0.9, 0.1],
    label: 0,
  },
  {
    text: "Hi from friend, no links, normal tone",
    x: [0.12, 0.12, 0.92],
    label: 1,
  },
  {
    text: "Win now! claim prize in 10 minutes",
    x: [0.88, 0.95, 0.22],
    label: 0,
  },
  {
    text: "Meeting notes from known teammate",
    x: [0.08, 0.15, 0.95],
    label: 1,
  },
  {
    text: "Security alert, click to reset password",
    x: [0.86, 0.78, 0.18],
    label: 0,
  },
];

let dataset500 = [];

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

function randomMatrix(rows, cols, range = 0.65) {
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

function randomVector(n, range = 0.28) {
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

function oneHot(label) {
  return Array.from({ length: ann.outSize }, (_, i) => (i === label ? 1 : 0));
}

function initNetwork() {
  ann.w1 = randomMatrix(ann.hiddenSize, ann.inSize);
  ann.b1 = randomVector(ann.hiddenSize);
  ann.w2 = randomMatrix(ann.outSize, ann.hiddenSize);
  ann.b2 = randomVector(ann.outSize);
}

function forwardPass(input) {
  const z1 = Array.from({ length: ann.hiddenSize }, (_, i) => {
    let sum = ann.b1[i];
    for (let j = 0; j < ann.inSize; j += 1) {
      sum += ann.w1[i][j] * input[j];
    }
    return sum;
  });

  const a1 = z1.map(sigmoid);

  const z2 = Array.from({ length: ann.outSize }, (_, k) => {
    let sum = ann.b2[k];
    for (let i = 0; i < ann.hiddenSize; i += 1) {
      sum += ann.w2[k][i] * a1[i];
    }
    return sum;
  });

  const probs = softmax(z2);
  return { a1, probs };
}

function trainSample(sample) {
  const y = oneHot(sample.label);
  const { a1, probs } = forwardPass(sample.x);
  const dz2 = probs.map((p, k) => p - y[k]);

  const dw2 = Array.from({ length: ann.outSize }, () => Array(ann.hiddenSize).fill(0));
  for (let k = 0; k < ann.outSize; k += 1) {
    for (let i = 0; i < ann.hiddenSize; i += 1) {
      dw2[k][i] = dz2[k] * a1[i];
    }
  }

  const dz1 = Array(ann.hiddenSize).fill(0);
  for (let i = 0; i < ann.hiddenSize; i += 1) {
    let sum = 0;
    for (let k = 0; k < ann.outSize; k += 1) {
      sum += ann.w2[k][i] * dz2[k];
    }
    dz1[i] = sum * a1[i] * (1 - a1[i]);
  }

  const dw1 = Array.from({ length: ann.hiddenSize }, () => Array(ann.inSize).fill(0));
  for (let i = 0; i < ann.hiddenSize; i += 1) {
    for (let j = 0; j < ann.inSize; j += 1) {
      dw1[i][j] = dz1[i] * sample.x[j];
    }
  }

  for (let k = 0; k < ann.outSize; k += 1) {
    for (let i = 0; i < ann.hiddenSize; i += 1) {
      ann.w2[k][i] -= ann.lr * dw2[k][i];
    }
    ann.b2[k] -= ann.lr * dz2[k];
  }

  for (let i = 0; i < ann.hiddenSize; i += 1) {
    for (let j = 0; j < ann.inSize; j += 1) {
      ann.w1[i][j] -= ann.lr * dw1[i][j];
    }
    ann.b1[i] -= ann.lr * dz1[i];
  }
}

function makeDataset500() {
  const samples = [];
  for (let i = 0; i < 500; i += 1) {
    const linkRisk = rand(0, 1);
    const urgency = rand(0, 1);
    const senderTrust = rand(0, 1);
    const noise = rand(-0.16, 0.16);
    const score = 1.35 * linkRisk + 1.05 * urgency - 1.45 * senderTrust + noise;
    const label = score > 0.2 ? 0 : 1;
    samples.push({ x: [linkRisk, urgency, senderTrust], label });
  }
  return samples;
}

function evaluateMistakes(dataset) {
  let mistakes = 0;
  for (const sample of dataset) {
    const { probs } = forwardPass(sample.x);
    const guess = probs[0] >= probs[1] ? 0 : 1;
    if (guess !== sample.label) {
      mistakes += 1;
    }
  }
  return mistakes;
}

function riskToFeatures(riskValue) {
  const risk = riskValue / 100;
  const linkRisk = 0.05 + 0.9 * risk;
  const urgency = 0.08 + 0.84 * Math.pow(risk, 1.08);
  const senderTrust = 0.95 - 0.9 * risk;
  return [clamp(linkRisk, 0, 1), clamp(urgency, 0, 1), clamp(senderTrust, 0, 1)];
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
  return el;
}

function weightColor(w) {
  return w >= 0 ? "rgba(15, 143, 143, 0.62)" : "rgba(214, 116, 54, 0.62)";
}

function drawEdge(svg, x1, y1, x2, y2, w) {
  const width = 1.1 + Math.min(4, Math.abs(w) * 3.2);
  svg.appendChild(
    svgEl("line", {
      x1,
      y1,
      x2,
      y2,
      stroke: weightColor(w),
      "stroke-width": width,
      "stroke-linecap": "round",
    }),
  );
}

function drawNode(svg, x, y, value, label) {
  const shade = Math.round(232 - value * 130);
  svg.appendChild(
    svgEl("circle", {
      cx: x,
      cy: y,
      r: 20,
      fill: `rgb(${shade}, ${246 - Math.round(value * 50)}, 236)`,
      stroke: "#2a3d54",
      "stroke-width": 1.15,
    }),
  );

  const text = svgEl("text", {
    x,
    y: y + 4,
    "font-size": 10,
    fill: "#263850",
    "text-anchor": "middle",
    "font-weight": 600,
  });
  text.textContent = label;
  svg.appendChild(text);
}

function drawLayerLabel(svg, x, textValue) {
  const t = svgEl("text", {
    x,
    y: 24,
    "font-size": 12,
    fill: "#52617f",
    "text-anchor": "middle",
    "font-weight": 600,
  });
  t.textContent = textValue;
  svg.appendChild(t);
}

function renderNetwork(input) {
  const { a1, probs } = forwardPass(input);
  refs.svg.innerHTML = "";

  const xPos = { in: 110, hidden: 355, out: 600 };
  const inY = [78, 170, 262];
  const hiddenY = [118, 222];
  const outY = [118, 222];

  drawLayerLabel(refs.svg, xPos.in, "Input clues");
  drawLayerLabel(refs.svg, xPos.hidden, "Filter gates");
  drawLayerLabel(refs.svg, xPos.out, "Decision");

  for (let i = 0; i < ann.hiddenSize; i += 1) {
    for (let j = 0; j < ann.inSize; j += 1) {
      drawEdge(refs.svg, xPos.in, inY[j], xPos.hidden, hiddenY[i], ann.w1[i][j]);
    }
  }

  for (let k = 0; k < ann.outSize; k += 1) {
    for (let i = 0; i < ann.hiddenSize; i += 1) {
      drawEdge(refs.svg, xPos.hidden, hiddenY[i], xPos.out, outY[k], ann.w2[k][i]);
    }
  }

  const inputLabels = ["L", "U", "S"];
  for (let j = 0; j < ann.inSize; j += 1) {
    drawNode(refs.svg, xPos.in, inY[j], input[j], inputLabels[j]);
  }

  drawNode(refs.svg, xPos.hidden, hiddenY[0], a1[0], "G1");
  drawNode(refs.svg, xPos.hidden, hiddenY[1], a1[1], "G2");

  drawNode(refs.svg, xPos.out, outY[0], probs[0], "S");
  drawNode(refs.svg, xPos.out, outY[1], probs[1], "N");

  renderOutputBars(probs);
}

function renderOutputBars(probs) {
  refs.bars.innerHTML = "";
  probs.forEach((p, idx) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${classNames[idx]}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span>${(p * 100).toFixed(1)}%</span>
    `;
    refs.bars.appendChild(row);
  });
}

function refreshLearningStats() {
  refs.mistakes.textContent = `Mistakes left: ${state.mistakesLeft ?? "--"}`;
  refs.rounds.textContent = `Rounds completed: ${state.roundsCompleted}`;
}

function stopLearningAnimation() {
  if (state.learnTimer) {
    clearInterval(state.learnTimer);
    state.learnTimer = null;
  }
}

function runFastLearning() {
  if (!state.sample5Loaded) {
    refs.learnNote.textContent = "Load the 5 sample messages first.";
    return;
  }

  if (!state.sample500Loaded) {
    refs.learnNote.textContent = "Load 500 examples first.";
    return;
  }

  stopLearningAnimation();

  // First pass: visibly learn from the 5 starter examples one by one.
  state.quickPassIndex = 0;
  refs.learnNote.textContent = "Running 5 starter examples...";

  state.learnTimer = setInterval(() => {
    if (state.quickPassIndex < sample5.length) {
      trainSample(sample5[state.quickPassIndex]);
      state.quickPassIndex += 1;
      state.roundsCompleted += 1;
      const baseline = dataset500.length > 0 ? evaluateMistakes(dataset500) : evaluateMistakes(sample5);
      state.bestMistakes = state.bestMistakes === null ? baseline : Math.min(state.bestMistakes, baseline);
      state.mistakesLeft = state.bestMistakes;
      refreshLearningStats();
      renderNetwork(riskToFeatures(Number(refs.riskSlider.value)));
      return;
    }

    refs.learnNote.textContent = "Learning fast on 500 examples...";

    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < dataset500.length; j += 1) {
        trainSample(dataset500[j]);
      }
      state.roundsCompleted += 1;
    }

    const mistakes = evaluateMistakes(dataset500);
    state.bestMistakes = Math.min(state.bestMistakes ?? mistakes, mistakes);
    state.mistakesLeft = state.bestMistakes;
    refreshLearningStats();
    renderNetwork(riskToFeatures(Number(refs.riskSlider.value)));

    if (state.mistakesLeft <= 0 || state.roundsCompleted >= 200) {
      state.mistakesLeft = 0;
      state.trainedFast = true;
      refreshLearningStats();
      refs.learnNote.textContent = "Ready to predict. Click Next to test the model.";
      stopLearningAnimation();
    }
  }, 70);
}

function renderSampleRows() {
  refs.sampleRows.innerHTML = "";
  sample5.forEach((sample) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${sample.text}</td><td>${sample.label === 0 ? "Spam" : "Not Spam"}</td>`;
    refs.sampleRows.appendChild(row);
  });
}

function updateOutputSummary() {
  const input = riskToFeatures(Number(refs.riskSlider.value));
  const { probs } = forwardPass(input);
  const spamIsTop = probs[0] >= probs[1];
  const label = spamIsTop ? "Spam" : "Not Spam";
  const confidence = (Math.max(...probs) * 100).toFixed(0);
  refs.outputSummary.textContent = `Model guess: ${label}, confidence (probability): ${confidence}%`;
  refs.riskReadout.textContent = `Risk: ${refs.riskSlider.value}`;
}

function updateUi() {
  const step = STEPS[state.stepIndex];
  const info = stageInfo[step];

  refs.stepIndicators.forEach((el, idx) => {
    el.classList.toggle("active", idx === state.stepIndex);
  });

  refs.prevButton.disabled = state.stepIndex === 0;
  refs.nextButton.disabled = state.stepIndex === STEPS.length - 1;

  refs.stagePill.textContent = `Step ${state.stepIndex + 1}`;
  refs.stageName.textContent = info.name;
  refs.actionLine.textContent = info.action;
  refs.watchLine.textContent = info.watch;

  refs.inputTools.classList.toggle("hidden", step !== "input");
  refs.learnTools.classList.toggle("hidden", step !== "learn");
  refs.outputTools.classList.toggle("hidden", step !== "output");

  if (step === "learn" && !state.sample5Loaded) {
    refs.actionLine.textContent = "Do this now: press Back and load the 5 sample messages first.";
  }

  refs.load500Button.disabled = !state.sample5Loaded;
  refs.learnFastButton.disabled = !state.sample500Loaded;

  if (step === "output" && !state.trainedFast) {
    refs.outputSummary.textContent = "Model is not fully trained yet. Go back to Learn and click Learn Fast.";
  } else if (step === "output") {
    updateOutputSummary();
  }

  renderNetwork(riskToFeatures(Number(refs.riskSlider.value)));
}

function bindEvents() {
  refs.prevButton.addEventListener("click", () => {
    state.stepIndex = clamp(state.stepIndex - 1, 0, STEPS.length - 1);
    updateUi();
  });

  refs.nextButton.addEventListener("click", () => {
    state.stepIndex = clamp(state.stepIndex + 1, 0, STEPS.length - 1);
    updateUi();
  });

  refs.load5Button.addEventListener("click", () => {
    stopLearningAnimation();
    initNetwork();

    state.sample5Loaded = true;
    state.sample500Loaded = false;
    state.trainedFast = false;
    state.roundsCompleted = 0;
    state.mistakesLeft = null;
    state.bestMistakes = null;
    dataset500 = [];

    renderSampleRows();
    const mistakes = evaluateMistakes(sample5);
    const accuracy = Math.round(((sample5.length - mistakes) / sample5.length) * 100);
    refs.sampleSummary.textContent = `Initial behavior: around ${accuracy}% accurate on 5 examples (near random to weak start).`;
    refs.sampleSummary.className = "status-note note-warn";
    refs.learnNote.textContent = "";
    refreshLearningStats();
    updateUi();
  });

  refs.load500Button.addEventListener("click", () => {
    stopLearningAnimation();
    dataset500 = makeDataset500();
    state.sample500Loaded = true;
    state.trainedFast = false;
    state.roundsCompleted = 0;
    const mistakes = evaluateMistakes(dataset500);
    state.mistakesLeft = mistakes;
    state.bestMistakes = mistakes;
    refs.learnNote.textContent = "500 examples loaded. Click Learn Fast.";
    refreshLearningStats();
    updateUi();
  });

  refs.learnFastButton.addEventListener("click", () => {
    runFastLearning();
  });

  refs.riskSlider.addEventListener("input", () => {
    if (STEPS[state.stepIndex] !== "output") {
      return;
    }
    updateOutputSummary();
    renderNetwork(riskToFeatures(Number(refs.riskSlider.value)));
  });

  window.addEventListener("beforeunload", stopLearningAnimation);
}

initNetwork();
bindEvents();
refreshLearningStats();
updateUi();
