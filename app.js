/* DS BG Remover PRO (GitHub Pages)
   - Modes: Auto, WhiteBG, U2NetP (Fast), U2Net Full, MODNet, Textile Hard Mask
   - Textile mode: hard binary alpha mask (embroidery-ready)
   - Auto white margin crop (optional, enabled by default)
   - Batch ZIP with JSZip
*/

const els = {
  file: document.getElementById("file"),
  files: document.getElementById("files"),
  srcCanvas: document.getElementById("srcCanvas"),
  outCanvas: document.getElementById("outCanvas"),
  btnRemove: document.getElementById("btnRemove"),
  btnDownload: document.getElementById("btnDownload"),
  btnBatch: document.getElementById("btnBatch"),
  status: document.getElementById("status"),
  perf: document.getElementById("perf"),
  btnDemo: document.getElementById("btnDemo"),

  mode: document.getElementById("mode"),
  batchType: document.getElementById("batchType"),

  alphaPower: document.getElementById("alphaPower"),
  alphaPowerVal: document.getElementById("alphaPowerVal"),
  refine: document.getElementById("refine"),
  refineVal: document.getElementById("refineVal"),
  feather: document.getElementById("feather"),
  featherVal: document.getElementById("featherVal"),

  whiteTol: document.getElementById("whiteTol"),
  whiteTolVal: document.getElementById("whiteTolVal"),
  whiteFeather: document.getElementById("whiteFeather"),
  whiteFeatherVal: document.getElementById("whiteFeatherVal"),

  maxW: document.getElementById("maxW"),
  maxWVal: document.getElementById("maxWVal"),
};

if (!window.ort) {
  console.error("onnxruntime-web (ort) not loaded.");
}

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";

const INPUT_SIZE = 320;

// ====== MODEL URLS (EDIT THIS) ======
// NOTE: u2netp small kept inside repo (fast)
// For u2net/modnet use stable URLs (GitHub Releases or HF resolve).
const MODEL_URLS = {
  u2netp: "./models/u2netp.onnx",

  // ✅ Replace these two with your working URLs:
  // GitHub Releases example:
  // u2net:  "https://github.com/<user>/<repo>/releases/download/v2/u2net.onnx",
  // modnet: "https://github.com/<user>/<repo>/releases/download/v2/model.onnx",

  // HuggingFace resolve example:
  u2net: "https://huggingface.co/dsccckadodara/ds-bg-models/resolve/main/u2net.onnx?download=true",
  modnet: "https://huggingface.co/dsccckadodara/ds-bg-models/resolve/main/model.onnx?download=true",
};

// ====== OPTIONS ======
const OPTS = {
  AUTO_CROP_WHITE_MARGIN: true, // ✅ auto crop white border before AI/white remove
  WHITE_MARGIN_THRESHOLD: 245,  // 245..250 recommended
  WHITE_MARGIN_PAD: 2,          // keep small padding around subject after crop
  TEXTILE_HARD_THRESHOLD: 0.55, // hard mask threshold (0.50..0.65)
};

// Sessions cache
const sessions = new Map();

let loadedImage = null;
let lastOutputBlob = null;
let lastBatchZipBlob = null;
let lastBatchCount = 0;

// ===== UI helpers =====
function setStatus(msg) { els.status.textContent = msg; }
function setPerf(msg) { els.perf.textContent = msg || ""; }

function updateLabels() {
  els.alphaPowerVal.textContent = els.alphaPower.value;
  els.refineVal.textContent = els.refine.value;
  els.featherVal.textContent = els.feather.value;
  els.whiteTolVal.textContent = els.whiteTol.value;
  els.whiteFeatherVal.textContent = els.whiteFeather.value;
  els.maxWVal.textContent = els.maxW.value;
}

["input", "change"].forEach(ev => {
  els.alphaPower.addEventListener(ev, updateLabels);
  els.refine.addEventListener(ev, updateLabels);
  els.feather.addEventListener(ev, updateLabels);
  els.whiteTol.addEventListener(ev, updateLabels);
  els.whiteFeather.addEventListener(ev, updateLabels);
  els.maxW.addEventListener(ev, updateLabels);
});
updateLabels();

// ===== Canvas / image helpers =====
function drawToCanvas(canvas, img, maxW) {
  const ctx = canvas.getContext("2d");
  const ratio = Math.min(1, maxW / img.width);
  canvas.width = Math.round(img.width * ratio);
  canvas.height = Math.max(1, Math.round(img.height * ratio));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}

function putImageDataToCanvas(canvas, imgData) {
  const ctx = canvas.getContext("2d");
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  ctx.putImageData(imgData, 0, 0);
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => {
    canvas.toBlob((b) => resolve(b), "image/png");
  });
}

function downloadBlob(blob, filename) {
  const a = document.createElement("a");
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function createImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = reject;
    img.src = url;
  });
}

// ===== Auto crop white margin (Textile) =====
function autoCropWhite(canvas, threshold = 245, pad = 2) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const w = canvas.width, h = canvas.height;
  if (w < 2 || h < 2) return;

  const data = ctx.getImageData(0, 0, w, h).data;

  let top = h, bottom = -1, left = w, right = -1;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const isWhite = (r > threshold && g > threshold && b > threshold);
      if (!isWhite) {
        if (y < top) top = y;
        if (y > bottom) bottom = y;
        if (x < left) left = x;
        if (x > right) right = x;
      }
    }
  }

  // If everything is white, skip
  if (bottom < 0 || right < 0) return;

  // padding
  top = Math.max(0, top - pad);
  left = Math.max(0, left - pad);
  bottom = Math.min(h - 1, bottom + pad);
  right = Math.min(w - 1, right + pad);

  const cropW = right - left + 1;
  const cropH = bottom - top + 1;

  if (cropW < 2 || cropH < 2) return;

  const cropped = ctx.getImageData(left, top, cropW, cropH);
  canvas.width = cropW;
  canvas.height = cropH;
  ctx.putImageData(cropped, 0, 0);
}

// ===== Mode detection =====
function isMostlyWhiteBackground(img) {
  const c = document.createElement("canvas");
  const w = Math.min(700, img.width);
  const ratio = w / img.width;
  c.width = w;
  c.height = Math.max(1, Math.round(img.height * ratio));
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, c.width, c.height);
  const data = ctx.getImageData(0, 0, c.width, c.height).data;

  let white = 0;
  const total = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 245 && data[i + 1] > 245 && data[i + 2] > 245) white++;
  }
  return (white / total) > 0.35;
}

// ===== White BG remover (Illustration) =====
function boxBlurAlpha(alpha, w, h, radius) {
  radius = Math.max(0, radius | 0);
  if (radius === 0) return alpha;

  const out = new Uint8ClampedArray(alpha.length);
  const tmp = new Uint32Array(alpha.length);

  // horizontal
  for (let y = 0; y < h; y++) {
    let sum = 0;
    for (let x = -radius; x <= radius; x++) {
      const xx = Math.min(w - 1, Math.max(0, x));
      sum += alpha[y * w + xx];
    }
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = sum;
      const x1 = x - radius;
      const x2 = x + radius + 1;
      if (x1 >= 0) sum -= alpha[y * w + x1];
      if (x2 < w) sum += alpha[y * w + x2];
    }
  }

  // vertical
  const win = radius * 2 + 1;
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -radius; y <= radius; y++) {
      const yy = Math.min(h - 1, Math.max(0, y));
      sum += tmp[yy * w + x];
    }
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / (win * win);
      const y1 = y - radius;
      const y2 = y + radius + 1;
      if (y1 >= 0) sum -= tmp[y1 * w + x];
      if (y2 < h) sum += tmp[y2 * w + x];
    }
  }
  return out;
}

function removeWhiteBGToImageData(canvas, tol = 238, feather = 2) {
  const w = canvas.width, h = canvas.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;

  const alpha = new Uint8ClampedArray(w * h);

  for (let i = 0; i < w * h; i++) {
    const r = d[i * 4 + 0], g = d[i * 4 + 1], b = d[i * 4 + 2];

    // Soft rule: treat only near-white as bg, preserve colors
    const isNearWhite = (r >= tol && g >= tol && b >= tol) &&
                        (Math.abs(r - g) < 12 && Math.abs(g - b) < 12);

    alpha[i] = isNearWhite ? 0 : 255;
  }

  const alpha2 = feather > 0 ? boxBlurAlpha(alpha, w, h, feather) : alpha;

  const out = new ImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    out.data[i * 4 + 0] = d[i * 4 + 0];
    out.data[i * 4 + 1] = d[i * 4 + 1];
    out.data[i * 4 + 2] = d[i * 4 + 2];
    out.data[i * 4 + 3] = alpha2[i];
  }
  return out;
}

// ===== AI helpers =====
function makeInputTensorFromImage(img) {
  const c = document.createElement("canvas");
  c.width = INPUT_SIZE; c.height = INPUT_SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const float = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);
  const HW = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < HW; i++) {
    float[i] = data[i * 4 + 0] / 255;
    float[i + HW] = data[i * 4 + 1] / 255;
    float[i + 2 * HW] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", float, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function normalizeMaskTo01(maskFloat) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < maskFloat.length; i++) {
    const v = maskFloat[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const denom = (max - min) || 1;
  const out = new Float32Array(maskFloat.length);
  for (let i = 0; i < maskFloat.length; i++) out[i] = (maskFloat[i] - min) / denom;
  return out;
}

function maskToAlphaSoft(mask01, alphaPower) {
  const exp = (alphaPower / 100); // ~0.6..2.4
  const out = new Uint8ClampedArray(mask01.length);
  for (let i = 0; i < mask01.length; i++) {
    const v = Math.max(0, Math.min(1, mask01[i]));
    out[i] = Math.pow(v, exp) * 255;
  }
  return out;
}

function maskToAlphaHard(mask01, threshold = 0.55) {
  const out = new Uint8ClampedArray(mask01.length);
  for (let i = 0; i < mask01.length; i++) out[i] = (mask01[i] > threshold) ? 255 : 0;
  return out;
}

function scaleMaskAlphaToSize(alpha320, w, h) {
  // Put alpha into 320 canvas then scale to w,h
  const mc = document.createElement("canvas");
  mc.width = INPUT_SIZE; mc.height = INPUT_SIZE;
  const mctx = mc.getContext("2d");
  const mimg = mctx.createImageData(INPUT_SIZE, INPUT_SIZE);

  for (let i = 0; i < alpha320.length; i++) {
    const a = alpha320[i];
    mimg.data[i * 4 + 0] = 255;
    mimg.data[i * 4 + 1] = 255;
    mimg.data[i * 4 + 2] = 255;
    mimg.data[i * 4 + 3] = a;
  }
  mctx.putImageData(mimg, 0, 0);

  const sc = document.createElement("canvas");
  sc.width = w; sc.height = h;
  const sctx = sc.getContext("2d");
  sctx.imageSmoothingEnabled = true;
  sctx.drawImage(mc, 0, 0, w, h);

  const scaled = sctx.getImageData(0, 0, w, h).data;
  const out = new Uint8ClampedArray(w * h);
  for (let i = 0; i < w * h; i++) out[i] = scaled[i * 4 + 3];
  return out;
}

function erodeDilate(alpha, w, h, amount) {
  amount = amount | 0;
  if (amount === 0) return alpha;

  const out = new Uint8ClampedArray(alpha.length);
  const r = Math.min(10, Math.abs(amount));
  const isDilate = amount > 0;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let best = isDilate ? 0 : 255;
      for (let yy = y - r; yy <= y + r; yy++) {
        const y2 = Math.min(h - 1, Math.max(0, yy));
        for (let xx = x - r; xx <= x + r; xx++) {
          const x2 = Math.min(w - 1, Math.max(0, xx));
          const v = alpha[y2 * w + x2];
          if (isDilate) { if (v > best) best = v; }
          else { if (v < best) best = v; }
        }
      }
      out[y * w + x] = best;
    }
  }
  return out;
}

function applyAlphaToCanvas(srcCanvas, alphaScaled) {
  const w = srcCanvas.width, h = srcCanvas.height;
  const ctx = srcCanvas.getContext("2d", { willReadFrequently: true });
  const src = ctx.getImageData(0, 0, w, h);
  const out = new ImageData(w, h);

  for (let i = 0; i < w * h; i++) {
    out.data[i * 4 + 0] = src.data[i * 4 + 0];
    out.data[i * 4 + 1] = src.data[i * 4 + 1];
    out.data[i * 4 + 2] = src.data[i * 4 + 2];
    out.data[i * 4 + 3] = alphaScaled[i];
  }
  return out;
}

// ===== ONNX session loader (URL DIRECT) =====
async function getSession(modelKey) {
  if (sessions.has(modelKey)) return sessions.get(modelKey);

  const url = MODEL_URLS[modelKey];
  if (!url) throw new Error(`Model URL missing for "${modelKey}" in MODEL_URLS`);

  setStatus(`Loading model: ${modelKey} …`);
  setPerf("");

  const t0 = performance.now();

  // ✅ Direct URL load (avoids fetch CORS issues)
  const session = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  const t1 = performance.now();
  sessions.set(modelKey, session);

  setStatus(`Model loaded ✅ (${modelKey})`);
  setPerf(`Model load ${modelKey}: ${(t1 - t0).toFixed(0)} ms`);
  return session;
}

async function runAI(modelKey, img) {
  const session = await getSession(modelKey);
  const input = makeInputTensorFromImage(img);

  const feeds = {};
  feeds[session.inputNames[0]] = input;

  const t0 = performance.now();
  const results = await session.run(feeds);
  const t1 = performance.now();

  const outName = session.outputNames[0];
  const maskFloat = results[outName].data;

  return { maskFloat, inferMs: (t1 - t0) };
}

// ===== Main processing =====
async function processSingleImage(img, mode) {
  const maxW = parseInt(els.maxW.value, 10);
  drawToCanvas(els.srcCanvas, img, maxW);

  if (OPTS.AUTO_CROP_WHITE_MARGIN) {
    autoCropWhite(els.srcCanvas, OPTS.WHITE_MARGIN_THRESHOLD, OPTS.WHITE_MARGIN_PAD);
  }

  const w = els.srcCanvas.width, h = els.srcCanvas.height;

  // AUTO logic
  if (mode === "auto") {
    if (isMostlyWhiteBackground(img)) mode = "white";
    else mode = "u2netp";
  }

  // WHITE MODE (Illustration / Motif)
  if (mode === "white") {
    const tol = parseInt(els.whiteTol.value, 10);
    const wf = parseInt(els.whiteFeather.value, 10);
    const out = removeWhiteBGToImageData(els.srcCanvas, tol, wf);
    putImageDataToCanvas(els.outCanvas, out);
    return { outBlob: await canvasToBlob(els.outCanvas), info: `WhiteBG tol=${tol}` };
  }

  // TEXTILE MODE = U2NET FULL + HARD BINARY MASK
  // (We will use U2Net Full model but export hard alpha)
  const isTextile = (mode === "textile");

  // Choose AI model key
  let modelKey = mode;
  if (isTextile) modelKey = "u2net"; // always full model

  const alphaPower = parseInt(els.alphaPower.value, 10);
  const refine = parseInt(els.refine.value, 10);
  const feather = parseInt(els.feather.value, 10);

  const { maskFloat, inferMs } = await runAI(modelKey, img);

  const mask01 = normalizeMaskTo01(maskFloat);

  // alpha 320
  let alpha320;
  if (isTextile) {
    alpha320 = maskToAlphaHard(mask01, OPTS.TEXTILE_HARD_THRESHOLD);
  } else {
    alpha320 = maskToAlphaSoft(mask01, alphaPower);
  }

  // scale to canvas size
  let alphaScaled = scaleMaskAlphaToSize(alpha320, w, h);

  // refine edges
  if (refine !== 0) alphaScaled = erodeDilate(alphaScaled, w, h, refine);

  // feather edges
  // For textile hard mask, by default keep feather 0 (user can still move slider)
  if (feather > 0) alphaScaled = boxBlurAlpha(alphaScaled, w, h, feather);

  const outImg = applyAlphaToCanvas(els.srcCanvas, alphaScaled);
  putImageDataToCanvas(els.outCanvas, outImg);

  const outBlob = await canvasToBlob(els.outCanvas);
  return { outBlob, info: `${isTextile ? "textile(u2net)" : modelKey} infer=${inferMs.toFixed(0)}ms` };
}

async function removeBackgroundMain() {
  els.btnRemove.disabled = true;
  els.btnDownload.disabled = true;
  els.btnBatch.disabled = true;
  lastOutputBlob = null;
  lastBatchZipBlob = null;
  lastBatchCount = 0;

  let mode = els.mode.value;

  try {
    const files = Array.from(els.files.files || []);
    const singleFile = els.file.files?.[0] || null;

    // BATCH
    if (files.length > 0) {
      setStatus(`Batch processing ${files.length} images…`);
      setPerf("");

      if (!window.JSZip) throw new Error("JSZip not loaded. Check jszip.min.js include.");

      const zip = new JSZip();
      const t0 = performance.now();

      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        setStatus(`Processing ${i + 1}/${files.length}: ${f.name}`);

        const img = await createImageFromFile(f);
        const { outBlob } = await processSingleImage(img, mode);

        const base = f.name.replace(/\.[^.]+$/, "");
        zip.file(`${base}_bg_removed.png`, outBlob);
      }

      const zipBlob = await zip.generateAsync({ type: "blob", compression: "DEFLATE" });
      const t1 = performance.now();

      lastBatchZipBlob = zipBlob;
      lastBatchCount = files.length;

      els.btnBatch.disabled = false;
      setStatus(`Batch done ✅ (${files.length} files)`);
      setPerf(`Total: ${(t1 - t0).toFixed(0)} ms`);
      return;
    }

    // SINGLE
    if (!singleFile && !loadedImage) {
      alert("Pehle single image upload karo (left input) ya batch files upload karo.");
      return;
    }

    const img = singleFile ? await createImageFromFile(singleFile) : loadedImage;
    setStatus("Processing…");
    setPerf("");

    const t0 = performance.now();
    const { outBlob, info } = await processSingleImage(img, mode);
    const t1 = performance.now();

    lastOutputBlob = outBlob;
    els.btnDownload.disabled = false;

    setStatus("Done ✅ Background removed.");
    setPerf(`${info} • total ${(t1 - t0).toFixed(0)} ms`);

  } catch (err) {
    console.error(err);
    setStatus("Error: " + (err?.message || err));
  } finally {
    els.btnRemove.disabled = false;
  }
}

// ===== Events =====
els.file.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  loadedImage = await createImageFromFile(file);

  const maxW = parseInt(els.maxW.value, 10);
  drawToCanvas(els.srcCanvas, loadedImage, maxW);

  if (OPTS.AUTO_CROP_WHITE_MARGIN) {
    autoCropWhite(els.srcCanvas, OPTS.WHITE_MARGIN_THRESHOLD, OPTS.WHITE_MARGIN_PAD);
  }

  // clear output
  const octx = els.outCanvas.getContext("2d");
  els.outCanvas.width = els.srcCanvas.width;
  els.outCanvas.height = els.srcCanvas.height;
  octx.clearRect(0, 0, els.outCanvas.width, els.outCanvas.height);

  lastOutputBlob = null;
  els.btnDownload.disabled = true;
  setStatus("Image loaded. Click Remove Background.");
});

els.files.addEventListener("change", () => {
  const count = (els.files.files || []).length;
  if (count > 0) setStatus(`Batch selected: ${count} files. Click Remove Background.`);
});

els.btnRemove.addEventListener("click", () => removeBackgroundMain());

els.btnDownload.addEventListener("click", () => {
  if (!lastOutputBlob) return;
  downloadBlob(lastOutputBlob, "bg-removed.png");
});

els.btnBatch.addEventListener("click", () => {
  if (!lastBatchZipBlob) return;
  downloadBlob(lastBatchZipBlob, `bg-removed_${lastBatchCount}_files.zip`);
});

els.btnDemo.addEventListener("click", async () => {
  const demoUrl = "https://images.unsplash.com/photo-1520975916090-3105956dac38?auto=format&fit=crop&w=900&q=80";
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    loadedImage = img;
    const maxW = parseInt(els.maxW.value, 10);
    drawToCanvas(els.srcCanvas, loadedImage, maxW);

    if (OPTS.AUTO_CROP_WHITE_MARGIN) {
      autoCropWhite(els.srcCanvas, OPTS.WHITE_MARGIN_THRESHOLD, OPTS.WHITE_MARGIN_PAD);
    }

    setStatus("Demo loaded. Click Remove Background.");
  };
  img.src = demoUrl;
});

setStatus("Ready. Upload image(s) → choose mode → Remove Background.");
