// app.js
/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Level 1: Standard reconstruction trap (pixel-wise MSE) -> copy input
 * Level 2: "Distribution" constraint (Sorted MSE idea) -> implemented as CDF-loss (soft histogram)
 * Level 3: Shape the geometry (Smoothness + Direction)
 *
 * Baseline: fixed model, MSE only.
 * Student: selectable architecture + custom loss with sliders.
 *
 * Fixes in this version:
 * - Robust slider binding with null checks + input/change events
 * - Clear console+log messages if slider elements are missing
 * - renderEvery defaults to 1 for immediate visual feedback
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.02,
  autoTrainSpeed: 50,
  renderEvery: 1, // show updates every step (makes slider effects easier to see)
};

const LOSS_COEFF = {
  dist: 1.0,
  smooth: 0.25,
  dir: 0.35,
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  optimizerBase: null,
  optimizerStudent: null,
};

// -------------------- Loss helpers --------------------

function mse(yTrue, yPred) {
  return tf.tidy(() => tf.mean(tf.square(yTrue.sub(yPred))));
}

function smoothness(yPred) {
  return tf.tidy(() => {
    const dx = yPred
      .slice([0, 0, 0, 0], [-1, -1, 15, -1])
      .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
    const dy = yPred
      .slice([0, 0, 0, 0], [-1, 15, -1, -1])
      .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
    return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
  });
}

function directionX(yPred) {
  return tf.tidy(() => {
    const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
    return tf.mean(yPred.mul(mask)).mul(-1);
  });
}

// Soft histogram + CDF loss (differentiable approximation of "Sorted MSE")
function softHistogram(xFlat, bins = 16, sigma = 0.04) {
  return tf.tidy(() => {
    const centers = tf.linspace(0, 1, bins); // [B]
    const x = xFlat.reshape([-1, 1]); // [N,1]
    const c = centers.reshape([1, -1]); // [1,B]
    const w = tf.exp(tf.neg(tf.square(x.sub(c)).div(2 * sigma * sigma))); // [N,B]
    const hist = tf.mean(w, 0); // [B]
    return hist.div(hist.sum().add(1e-8));
  });
}

function cdfLoss(yTrue, yPred, bins = 16, sigma = 0.04) {
  return tf.tidy(() => {
    const a = yTrue.reshape([-1]);
    const b = yPred.reshape([-1]);
    const ha = softHistogram(a, bins, sigma);
    const hb = softHistogram(b, bins, sigma);
    const cdfa = tf.cumsum(ha);
    const cdfb = tf.cumsum(hb);
    return tf.mean(tf.square(cdfa.sub(cdfb)));
  });
}

// Student custom loss: Level2 + Level3
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const Ldist = cdfLoss(yTrue, yPred).mul(LOSS_COEFF.dist);
    const Lsmooth = smoothness(yPred).mul(LOSS_COEFF.smooth);
    const Ldir = directionX(yPred).mul(LOSS_COEFF.dir);
    return Ldist.add(Lsmooth).add(Ldir);
  });
}

// -------------------- Models --------------------

function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// -------------------- Training loop --------------------

function trainModelStep(model, optimizer, lossFn, x) {
  const varList = model.trainableWeights.map((w) => w.val);

  const { value, grads } = tf.variableGrads(() => {
    const yPred = model.apply(x);
    return lossFn(x, yPred);
  }, varList);

  optimizer.applyGradients(grads);

  const lossVal = value.dataSync()[0];
  value.dispose();
  Object.values(grads).forEach((g) => g.dispose());
  return lossVal;
}

async function trainStep() {
  state.step += 1;

  try {
    const baseLoss = tf.tidy(() =>
      trainModelStep(state.baselineModel, state.optimizerBase, mse, state.xInput),
    );
    const studLoss = tf.tidy(() =>
      trainModelStep(
        state.studentModel,
        state.optimizerStudent,
        studentLoss,
        state.xInput,
      ),
    );

    if (state.step % CONFIG.renderEvery === 0 || !state.isAutoTraining) {
      await render();
      updateLossDisplay(baseLoss, studLoss);
    }

    log(
      `Step ${state.step}: Base=${baseLoss.toFixed(4)} | Student=${studLoss.toFixed(
        4,
      )} | λDist=${LOSS_COEFF.dist.toFixed(2)} λSmooth=${LOSS_COEFF.smooth.toFixed(
        2,
      )} λDir=${LOSS_COEFF.dir.toFixed(2)}`,
    );
  } catch (e) {
    log(`Training error: ${e.message}`, true);
    stopAutoTrain();
  }
}

// -------------------- UI helpers --------------------

async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    basePred.squeeze(),
    document.getElementById("canvas-baseline"),
  );
  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student"),
  );

  basePred.dispose();
  studPred.dispose();
}

function updateLossDisplay(base, stud) {
  const b = document.getElementById("loss-baseline");
  const s = document.getElementById("loss-student");
  if (b) b.innerText = `Loss: ${base.toFixed(5)}`;
  if (s) s.innerText = `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  if (!el) {
    console[isError ? "error" : "log"](msg);
    return;
  }
  const div = document.createElement("div");
  div.innerText = `> ${msg}`;
  if (isError) div.classList.add("error");
  el.prepend(div);
}

function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    stopAutoTrain();
    return;
  }
  state.isAutoTraining = true;
  if (btn) {
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
  }
  loop();
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  if (btn) {
    btn.innerText = "Auto Train (Start)";
    btn.classList.add("btn-auto");
    btn.classList.remove("btn-stop");
  }
}

function loop() {
  if (!state.isAutoTraining) return;
  trainStep();
  setTimeout(loop, CONFIG.autoTrainSpeed);
}

function resetModels(archType = null) {
  if (state.isAutoTraining) stopAutoTrain();

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizerBase) state.optimizerBase.dispose();
  if (state.optimizerStudent) state.optimizerStudent.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);

  state.optimizerBase = tf.train.adam(CONFIG.learningRate);
  state.optimizerStudent = tf.train.adam(CONFIG.learningRate);

  state.step = 0;
  log(`Models reset. Student Arch: ${archType}`);
  render();
}

// Robust slider binding (fix for your issue)
function bindSlider(id, valId, key) {
  const s = document.getElementById(id);
  const v = document.getElementById(valId);

  if (!s || !v) {
    log(`Slider bind failed: ${id} or ${valId} not found`, true);
    return;
  }

  const set = () => {
    LOSS_COEFF[key] = parseFloat(s.value);
    v.textContent = LOSS_COEFF[key].toFixed(2);
  };

  s.addEventListener("input", set);
  s.addEventListener("change", set);
  set();
}

function init() {
  // fixed input noise in [0,1]
  state.xInput = tf.randomUniform(CONFIG.inputShapeData, 0, 1, "float32");

  // draw input once
  const cIn = document.getElementById("canvas-input");
  if (cIn) tf.browser.toPixels(state.xInput.squeeze(), cIn);

  // init models
  resetModels("compression");

  // buttons
  const btnTrain = document.getElementById("btn-train");
  const btnAuto = document.getElementById("btn-auto");
  const btnReset = document.getElementById("btn-reset");

  if (btnTrain) btnTrain.addEventListener("click", trainStep);
  if (btnAuto) btnAuto.addEventListener("click", toggleAutoTrain);
  if (btnReset)
    btnReset.addEventListener("click", () => {
      const checked = document.querySelector('input[name="arch"]:checked');
      const arch = checked ? checked.value : "compression";
      resetModels(arch);
    });

  // architecture radios
  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      const arch = e.target.value;
      const lbl = document.getElementById("student-arch-label");
      if (lbl) lbl.innerText = arch.charAt(0).toUpperCase() + arch.slice(1);
      resetModels(arch);
    });
  });

  // sliders
  bindSlider("sld-dist", "val-dist", "dist");
  bindSlider("sld-smooth", "val-smooth", "smooth");
  bindSlider("sld-dir", "val-dir", "dir");

  log("Initialized. Ready to train.");
  log(
    `If sliders don't update: hard refresh (Ctrl+F5) and check that index.html contains the slider elements.`,
  );
}

init();
