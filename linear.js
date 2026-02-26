const STEPS = ["input", "learn", "output"];

const state = {
  stepIndex: 0,
  dataLoaded: false,
  lineFitted: false,
  fitTimer: null,
  points: [],
  m: 0,
  b: 0,
  targetM: 0,
  targetB: 0,
  rounds: 0,
};

const refs = {
  stepIndicators: [...document.querySelectorAll(".step-indicator")],
  prevButton: document.getElementById("prev-step"),
  nextButton: document.getElementById("next-step"),
  stagePill: document.getElementById("stage-pill"),
  stageName: document.getElementById("stage-name"),
  actionLine: document.getElementById("action-line"),
  watchLine: document.getElementById("watch-line"),
  svg: document.getElementById("linear-svg"),
  loadButton: document.getElementById("load-points"),
  fitButton: document.getElementById("fit-line"),
  predictControl: document.getElementById("predict-control"),
  xSlider: document.getElementById("linear-x-slider"),
  xReadout: document.getElementById("linear-x-readout"),
  feedback: document.getElementById("linear-feedback"),
};

const stageInfo = {
  input: {
    name: "Input",
    action: "Do this now: click Load Example Points.",
    watch: "What to watch: a cloud of noisy dots appears.",
  },
  learn: {
    name: "Learn",
    action: "Do this now: click Fit Trend Line.",
    watch: "What to watch: the line moves into the best position.",
  },
  output: {
    name: "Output",
    action: "Do this now: move the x slider to test predictions.",
    watch: "What to watch: the orange point shows the predicted value.",
  },
};

const chart = {
  w: 700,
  h: 360,
  margin: { top: 18, right: 22, bottom: 42, left: 58 },
};

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

function scaleX(x) {
  const { w, margin } = chart;
  return margin.left + (x / 10) * (w - margin.left - margin.right);
}

function scaleY(y) {
  const { h, margin } = chart;
  return h - margin.bottom - (y / 20) * (h - margin.top - margin.bottom);
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
  return el;
}

function makePoints() {
  const trueM = rand(0.68, 1.45);
  const trueB = rand(2.2, 6.2);
  const points = [];
  for (let i = 0; i < 26; i += 1) {
    const x = rand(0.5, 9.5);
    const noise = rand(-1.3, 1.3);
    const y = clamp(trueM * x + trueB + noise, 0.4, 19.2);
    points.push({ x, y });
  }
  return points;
}

function calcBestFit(points) {
  const n = points.length;
  let sx = 0;
  let sy = 0;
  let sxy = 0;
  let sxx = 0;

  for (const p of points) {
    sx += p.x;
    sy += p.y;
    sxy += p.x * p.y;
    sxx += p.x * p.x;
  }

  const denominator = n * sxx - sx * sx;
  if (Math.abs(denominator) < 1e-9) {
    return { m: 0, b: sy / n };
  }

  const m = (n * sxy - sx * sy) / denominator;
  const b = (sy - m * sx) / n;
  return { m, b };
}

function predict(x) {
  return state.m * x + state.b;
}

function stopFitAnimation() {
  if (state.fitTimer) {
    clearInterval(state.fitTimer);
    state.fitTimer = null;
  }
}

function startFitAnimation() {
  if (!state.dataLoaded) {
    refs.feedback.textContent = "Load points in Step 1 first, then return to Step 2.";
    return;
  }

  stopFitAnimation();
  state.rounds = 0;
  state.m = rand(-0.4, 0.5);
  state.b = rand(7, 13);
  state.lineFitted = false;
  renderLinear();

  state.fitTimer = setInterval(() => {
    state.m += (state.targetM - state.m) * 0.12;
    state.b += (state.targetB - state.b) * 0.12;
    state.rounds += 1;

    if (state.rounds >= 90 || (Math.abs(state.targetM - state.m) < 0.01 && Math.abs(state.targetB - state.b) < 0.03)) {
      state.m = state.targetM;
      state.b = state.targetB;
      state.lineFitted = true;
      stopFitAnimation();
      refs.feedback.textContent = "Line fitted. Go Next to test predictions.";
    } else {
      refs.feedback.textContent = `Fitting line... ${Math.min(99, Math.floor((state.rounds / 90) * 100))}%`;
    }

    renderLinear();
  }, 45);
}

function drawAxes(svg) {
  const g = svgEl("g", { opacity: "0.95" });
  const { w, h, margin } = chart;

  g.appendChild(svgEl("line", { x1: margin.left, y1: h - margin.bottom, x2: w - margin.right, y2: h - margin.bottom, stroke: "#8fa2ba", "stroke-width": 1.4 }));
  g.appendChild(svgEl("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: h - margin.bottom, stroke: "#8fa2ba", "stroke-width": 1.4 }));

  for (let x = 0; x <= 10; x += 2) {
    const px = scaleX(x);
    g.appendChild(svgEl("line", { x1: px, y1: h - margin.bottom, x2: px, y2: h - margin.bottom + 7, stroke: "#8fa2ba", "stroke-width": 1 }));
    const t = svgEl("text", { x: px, y: h - margin.bottom + 21, fill: "#5a6b86", "font-size": 11, "text-anchor": "middle" });
    t.textContent = String(x);
    g.appendChild(t);
  }

  for (let y = 0; y <= 20; y += 5) {
    const py = scaleY(y);
    g.appendChild(svgEl("line", { x1: margin.left - 7, y1: py, x2: margin.left, y2: py, stroke: "#8fa2ba", "stroke-width": 1 }));
    const t = svgEl("text", { x: margin.left - 12, y: py + 4, fill: "#5a6b86", "font-size": 11, "text-anchor": "end" });
    t.textContent = String(y);
    g.appendChild(t);
  }

  const xLabel = svgEl("text", { x: w - margin.right, y: h - 10, fill: "#52617f", "font-size": 11, "text-anchor": "end" });
  xLabel.textContent = "Input x";
  g.appendChild(xLabel);

  const yLabel = svgEl("text", { x: 18, y: margin.top - 2, fill: "#52617f", "font-size": 11 });
  yLabel.textContent = "Output y";
  g.appendChild(yLabel);

  svg.appendChild(g);
}

function renderLinear() {
  refs.svg.innerHTML = "";
  drawAxes(refs.svg);

  if (state.dataLoaded) {
    const group = svgEl("g");
    state.points.forEach((p) => {
      group.appendChild(svgEl("circle", { cx: scaleX(p.x), cy: scaleY(p.y), r: 4.5, fill: "#c0575f", opacity: 0.82 }));
    });
    refs.svg.appendChild(group);
  }

  const step = STEPS[state.stepIndex];
  if (step !== "input" && state.dataLoaded) {
    const line = svgEl("line", {
      x1: scaleX(0),
      y1: scaleY(clamp(predict(0), 0, 20)),
      x2: scaleX(10),
      y2: scaleY(clamp(predict(10), 0, 20)),
      stroke: "#0f8f8f",
      "stroke-width": 3,
      "stroke-linecap": "round",
    });
    refs.svg.appendChild(line);
  }

  if (step === "output" && state.lineFitted) {
    const x = Number(refs.xSlider.value);
    const y = clamp(predict(x), 0, 20);
    refs.svg.appendChild(svgEl("line", { x1: scaleX(x), y1: scaleY(0), x2: scaleX(x), y2: scaleY(y), stroke: "#d67436", "stroke-width": 2, "stroke-dasharray": "6 6" }));
    refs.svg.appendChild(svgEl("circle", { cx: scaleX(x), cy: scaleY(y), r: 6, fill: "#d67436" }));
  }

  refs.xReadout.textContent = `x = ${Number(refs.xSlider.value).toFixed(1)}`;
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

  refs.loadButton.classList.toggle("hidden", step !== "input");
  refs.fitButton.classList.toggle("hidden", step !== "learn");
  refs.predictControl.classList.toggle("hidden", step !== "output");

  if (step === "learn" && !state.dataLoaded) {
    refs.fitButton.disabled = true;
    refs.actionLine.textContent = "Do this now: press Back, then load example points in Step 1.";
    refs.feedback.textContent = "Data not loaded yet.";
  } else {
    refs.fitButton.disabled = false;
    if (step === "input" && !state.dataLoaded) {
      refs.feedback.textContent = "Start by loading points.";
    } else if (step === "learn" && state.dataLoaded && !state.lineFitted && !state.fitTimer) {
      refs.feedback.textContent = "Ready to fit. Click Fit Trend Line.";
    } else if (step === "output" && !state.lineFitted) {
      refs.feedback.textContent = "Go back to Step 2 and fit the trend line first.";
    } else if (step === "output" && state.lineFitted) {
      const pred = predict(Number(refs.xSlider.value));
      refs.feedback.textContent = `Prediction: y is about ${pred.toFixed(1)}.`;
    }
  }

  refs.xSlider.disabled = !state.lineFitted;
  renderLinear();
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

  refs.loadButton.addEventListener("click", () => {
    state.points = makePoints();
    const bestFit = calcBestFit(state.points);
    state.targetM = bestFit.m;
    state.targetB = bestFit.b;
    state.dataLoaded = true;
    state.lineFitted = false;
    stopFitAnimation();
    refs.feedback.textContent = "Points loaded. Click Next to continue.";
    updateUi();
  });

  refs.fitButton.addEventListener("click", () => {
    startFitAnimation();
  });

  refs.xSlider.addEventListener("input", () => {
    if (!state.lineFitted) {
      return;
    }
    const pred = predict(Number(refs.xSlider.value));
    refs.feedback.textContent = `Prediction: y is about ${pred.toFixed(1)}.`;
    renderLinear();
  });

  window.addEventListener("beforeunload", stopFitAnimation);
}

bindEvents();
updateUi();
