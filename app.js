// app.js
/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Goal:
 * - Baseline model: MSE-only (copies the input noise).
 * - Student model: same starting point, but loss adds "intent":
 *   smoothness + left->right direction so structure emerges.
 *
 * This file is written to be minimal and readable for students.
 */

// ==========================================
// 1) Global Config + State
// ==========================================
const CONFIG = {
  inputShapeModel: [16, 16, 1], // for layer inputShape (no batch)
  inputShapeData: [1, 16, 16, 1], // actual tensor shape (with batch)
  learningRate: 0.02,
  autoTrainSpeed: 50, // ms between steps
  renderEvery: 5,
};

// Student loss coefficients (tune these)
const LAMBDA_SMOOTH = 0.25; // ↑ more smoothing
const LAMBDA_DIR = 0.35; // ↑ more left-dark/right-bright push

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  optimizerBase: null,
  optimizerStudent: null,
};

// ==========================================
// 2) Loss helpers (given to students)
// ==========================================

// MSE -> scalar
function mse(yTrue, yPred) {
  return tf.tidy(() => tf.mean(tf.square(yTrue.sub(yPred))));
}

// Smoothness (total-variation style): penalize neighbor differences
function smoothness(yPred) {
  return tf.tidy(() => {
    // yPred shape: [1, 16, 16, 1]
    const dx = yPred
      .slice([0, 0, 0, 0], [-1, -1, 15, -1])
      .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
    const dy = yPred
      .slice([0, 0, 0, 0], [-1, 15, -1, -1])
      .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
    return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
  });
}

// DirectionX: encourage brighter pixels on the right than on the left
function directionX(yPred) {
  return tf.tidy(() => {
    // mask from -1 (left) to +1 (right)
    const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]); // [1,1,16,1]
    // maximize mean(yPred * mask) => minimize negative
    return tf.mean(yPred.mul(mask)).mul(-1);
  });
}

// ==========================================
// 3) Model architectures
// ==========================================

function createBaselineModel() {
  // Fixed Compression AE: 256 -> 64 -> 256
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ------------------------------------------------------------------
// TODO-A (Architecture): student selectable projection type
// - compression: implemented
// - transformation: implemented (256 -> 256 -> 256)
// - expansion: implemented (256 -> 512 -> 256)
// ------------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    // 1:1-ish projection: keep dimension
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    // overcomplete projection: expand then project back
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4) Student custom loss (the "intent")
// ==========================================

// ------------------------------------------------------------------
// TODO-B (Custom loss): students usually edit coefficients/terms here
// Baseline: MSE only
// Student:  MSE + smoothness + directionX
//
// Note about "no new colors":
// - Strict histogram preservation is hard with dense nets,
//   so we approximate the "sliding puzzle" constraint by keeping MSE
//   strong enough to anchor values near the input distribution.
// ------------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossMSE = mse(yTrue, yPred);
    const lossSmooth = smoothness(yPred).mul(LAMBDA_SMOOTH);
    const lossDir = directionX(yPred).mul(LAMBDA_DIR);

    // Total
    return lossMSE.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// 5) Training (custom loop, no model.fit)
// ==========================================

function trainModelStep(model, optimizer, lossFn, x) {
  // Use trainable variables (NOT getWeights())
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

  let baseLoss = 0;
  let studLoss = 0;

  try {
    baseLoss = tf.tidy(() =>
      trainModelStep(state.baselineModel, state.optimizerBase, mse, state.xInput),
    );

    studLoss = tf.tidy(() =>
      trainModelStep(
        state.studentModel,
        state.optimizerStudent,
        studentLoss,
        state.xInput,
      ),
    );

    // TODO-C (Comparison): show both losses + visual difference
    log(
      `Step ${state.step}: Base Loss=${baseLoss.toFixed(4)} | Student Loss=${studLoss.toFixed(4)}`,
    );

    if (state.step % CONFIG.renderEvery === 0 || !state.isAutoTraining) {
      await render();
      updateLossDisplay(baseLoss, studLoss);
    }
  } catch (e) {
    log(`Training error: ${e.message}`, true);
    stopAutoTrain();
  }
}

// ==========================================
// 6) Rendering + UI
// ==========================================

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
  document.getElementById("loss-baseline").innerText = `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText = `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
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
  btn.innerText = "Auto Train (Stop)";
  btn.classList.add("btn-stop");
  btn.classList.remove("btn-auto");
  loop();
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Auto Train (Start)";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
}

function loop() {
  if (!state.isAutoTraining) return;
  trainStep();
  setTimeout(loop, CONFIG.autoTrainSpeed);
}

// Reset + init
function resetModels(archType = null) {
  if (state.isAutoTraining) stopAutoTrain();

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  // Dispose old
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizerBase) state.optimizerBase.dispose();
  if (state.optimizerStudent) state.optimizerStudent.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);

  // Separate optimizers (clean separation, easier for students)
  state.optimizerBase = tf.train.adam(CONFIG.learningRate);
  state.optimizerStudent = tf.train.adam(CONFIG.learningRate);

  state.step = 0;
  log(`Models reset. Student Arch: ${archType}`);
  render();
}

function init() {
  // fixed input noise in [0,1]
  state.xInput = tf.randomUniform(CONFIG.inputShapeData, 0, 1, "float32");

  // draw input once
  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input"),
  );

  // init models
  resetModels("compression");

  // bind UI
  document.getElementById("btn-train").addEventListener("click", trainStep);
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", () => {
    const checked = document.querySelector('input[name="arch"]:checked');
    const arch = checked ? checked.value : "compression";
    resetModels(arch);
  });

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      const arch = e.target.value;
      document.getElementById("student-arch-label").innerText =
        arch.charAt(0).toUpperCase() + arch.slice(1);
      resetModels(arch);
    });
  });

  log("Initialized. Ready to train.");
}

init();
