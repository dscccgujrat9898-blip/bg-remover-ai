/* DS BG Remover (GitHub Pages) - U2NetP + onnxruntime-web
   Model: ./models/u2netp.onnx (keep inside repo)
*/

const els = {
  file: document.getElementById("file"),
  srcCanvas: document.getElementById("srcCanvas"),
  outCanvas: document.getElementById("outCanvas"),
  btnRemove: document.getElementById("btnRemove"),
  btnDownload: document.getElementById("btnDownload"),
  status: document.getElementById("status"),
  perf: document.getElementById("perf"),
  threshold: document.getElementById("threshold"),
  blur: document.getElementById("blur"),
  feather: document.getElementById("feather"),
  thresholdVal: document.getElementById("thresholdVal"),
  blurVal: document.getElementById("blurVal"),
  featherVal: document.getElementById("featherVal"),
  btnDemo: document.getElementById("btnDemo"),
};

let session = null;
let lastOutputBlob = null;
let loadedImage = null;

const MODEL_PATH = "./models/u2netp.onnx";
const INPUT_SIZE = 320;

// IMPORTANT: wasm path set (CDN). Official docs recommend wasmPaths config. :contentReference[oaicite:4]{index=4}
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/"; // keep versions consistent

function setStatus(msg){ els.status.textContent = msg; }
function setPerf(msg){ els.perf.textContent = msg || ""; }

function updateSliderLabels(){
  els.thresholdVal.textContent = els.threshold.value;
  els.blurVal.textContent = els.blur.value;
  els.featherVal.textContent = els.feather.value;
}
["input","change"].forEach(ev=>{
  els.threshold.addEventListener(ev, updateSliderLabels);
  els.blur.addEventListener(ev, updateSliderLabels);
  els.feather.addEventListener(ev, updateSliderLabels);
});
updateSliderLabels();

async function loadModel(){
  if (session) return session;
  setStatus("Loading model… (first time may take a few seconds)");
  setPerf("");
  const t0 = performance.now();

  session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  const t1 = performance.now();
  setStatus("Model loaded ✅ Now upload an image.");
  setPerf(`Model load: ${(t1 - t0).toFixed(0)} ms`);
  return session;
}

function drawToCanvas(canvas, img, maxW = 1200){
  const ctx = canvas.getContext("2d");
  const ratio = Math.min(1, maxW / img.width);
  canvas.width = Math.round(img.width * ratio);
  canvas.height = Math.round(img.height * ratio);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}

function createImageFromFile(file){
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = reject;
    img.src = url;
  });
}

function makeInputTensorFromImage(img){
  // Draw image to temp canvas at 320x320
  const c = document.createElement("canvas");
  c.width = INPUT_SIZE;
  c.height = INPUT_SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  // U2Net expects float32 CHW normalized 0..1
  const float = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);
  let p = 0;
  const HW = INPUT_SIZE * INPUT_SIZE;

  for (let i = 0; i < HW; i++) {
    const r = data[i*4 + 0] / 255;
    const g = data[i*4 + 1] / 255;
    const b = data[i*4 + 2] / 255;
    float[i] = r;          // C0
    float[i + HW] = g;     // C1
    float[i + 2*HW] = b;   // C2
  }

  return new ort.Tensor("float32", float, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function normalizeMask(mask){
  // mask: Float32Array length 320*320
  // normalize to 0..255
  let min = Infinity, max = -Infinity;
  for (let i=0;i<mask.length;i++){
    const v = mask[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const out = new Uint8ClampedArray(mask.length);
  const denom = (max - min) || 1;
  for (let i=0;i<mask.length;i++){
    out[i] = Math.max(0, Math.min(255, ((mask[i] - min) / denom) * 255));
  }
  return out;
}

function boxBlurAlpha(alpha, w, h, radius){
  radius = Math.max(0, radius|0);
  if (radius === 0) return alpha;
  const out = new Uint8ClampedArray(alpha.length);
  const tmp = new Uint32Array(alpha.length);

  // horizontal
  for (let y=0;y<h;y++){
    let sum = 0;
    for (let x=-radius;x<=radius;x++){
      const xx = Math.min(w-1, Math.max(0, x));
      sum += alpha[y*w + xx];
    }
    for (let x=0;x<w;x++){
      tmp[y*w + x] = sum;
      const x1 = x - radius;
      const x2 = x + radius + 1;
      if (x1 >= 0) sum -= alpha[y*w + x1];
      if (x2 < w) sum += alpha[y*w + x2];
    }
  }

  // vertical
  const win = radius*2 + 1;
  for (let x=0;x<w;x++){
    let sum = 0;
    for (let y=-radius;y<=radius;y++){
      const yy = Math.min(h-1, Math.max(0, y));
      sum += tmp[yy*w + x];
    }
    for (let y=0;y<h;y++){
      const v = sum / (win*win);
      out[y*w + x] = v;
      const y1 = y - radius;
      const y2 = y + radius + 1;
      if (y1 >= 0) sum -= tmp[y1*w + x];
      if (y2 < h) sum += tmp[y2*w + x];
    }
  }
  return out;
}

function featherAlpha(alpha, w, h, amount){
  amount = Math.max(0, amount|0);
  if (amount === 0) return alpha;
  // simple: blur a bit more to soften edges
  return boxBlurAlpha(alpha, w, h, Math.ceil(amount/2));
}

function applyMaskToOriginal(originalCanvas, mask320, threshold, blur, feather){
  // mask320: Uint8ClampedArray length 320*320 (0..255)
  // We scale mask to original canvas size
  const srcCtx = originalCanvas.getContext("2d", { willReadFrequently: true });
  const w = originalCanvas.width;
  const h = originalCanvas.height;
  const srcImg = srcCtx.getImageData(0,0,w,h);
  const src = srcImg.data;

  // Create mask canvas to scale
  const mc = document.createElement("canvas");
  mc.width = INPUT_SIZE;
  mc.height = INPUT_SIZE;
  const mctx = mc.getContext("2d");
  const mimg = mctx.createImageData(INPUT_SIZE, INPUT_SIZE);
  for (let i=0;i<mask320.length;i++){
    const v = mask320[i];
    mimg.data[i*4+0] = v;
    mimg.data[i*4+1] = v;
    mimg.data[i*4+2] = v;
    mimg.data[i*4+3] = 255;
  }
  mctx.putImageData(mimg,0,0);

  // Scale mask to original size
  const sc = document.createElement("canvas");
  sc.width = w;
  sc.height = h;
  const sctx = sc.getContext("2d");
  sctx.imageSmoothingEnabled = true;
  sctx.drawImage(mc, 0,0,w,h);
  const scaled = sctx.getImageData(0,0,w,h).data;

  // Extract alpha, apply threshold
  let alpha = new Uint8ClampedArray(w*h);
  for (let i=0;i<w*h;i++){
    const v = scaled[i*4]; // grayscale
    alpha[i] = (v >= threshold) ? 255 : 0;
  }

  // Blur/Feather
  alpha = boxBlurAlpha(alpha, w, h, blur);
  alpha = featherAlpha(alpha, w, h, feather);

  // Compose output RGBA
  const out = new ImageData(w,h);
  const dst = out.data;

  for (let i=0;i<w*h;i++){
    const a = alpha[i];
    dst[i*4+0] = src[i*4+0];
    dst[i*4+1] = src[i*4+1];
    dst[i*4+2] = src[i*4+2];
    dst[i*4+3] = a;
  }

  return out;
}

async function removeBackground(){
  if (!loadedImage) {
    alert("Pehle image upload karo.");
    return;
  }
  await loadModel();

  els.btnRemove.disabled = true;
  els.btnDownload.disabled = true;
  setStatus("Processing…");
  setPerf("");

  const t0 = performance.now();

  // Build tensor
  const input = makeInputTensorFromImage(loadedImage);

  // Run
  const feeds = {};
  // Get input name dynamically
  const inputName = session.inputNames[0];
  feeds[inputName] = input;

  const results = await session.run(feeds);
  // Output name may vary; take first output
  const outName = session.outputNames[0];
  const outTensor = results[outName];

  // outTensor: [1,1,320,320] or [1,320,320]
  const data = outTensor.data;
  const mask = normalizeMask(data);

  // Apply to original canvas (the displayed size)
  const threshold = parseInt(els.threshold.value, 10);
  const blur = parseInt(els.blur.value, 10);
  const feather = parseInt(els.feather.value, 10);

  const composed = applyMaskToOriginal(els.srcCanvas, mask, threshold, blur, feather);

  // Draw result
  const octx = els.outCanvas.getContext("2d");
  els.outCanvas.width = els.srcCanvas.width;
  els.outCanvas.height = els.srcCanvas.height;
  octx.putImageData(composed, 0, 0);

  // Enable download
  els.outCanvas.toBlob((blob)=>{
    lastOutputBlob = blob;
    els.btnDownload.disabled = !blob;
  }, "image/png");

  const t1 = performance.now();
  setStatus("Done ✅ Background removed.");
  setPerf(`Inference + compose: ${(t1 - t0).toFixed(0)} ms`);

  els.btnRemove.disabled = false;
}

function downloadPNG(){
  if (!lastOutputBlob) return;
  const a = document.createElement("a");
  const url = URL.createObjectURL(lastOutputBlob);
  a.href = url;
  a.download = "bg-removed.png";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Events
els.file.addEventListener("change", async (e)=>{
  const file = e.target.files?.[0];
  if (!file) return;
  loadedImage = await createImageFromFile(file);
  drawToCanvas(els.srcCanvas, loadedImage);
  // clear output
  const octx = els.outCanvas.getContext("2d");
  els.outCanvas.width = els.srcCanvas.width;
  els.outCanvas.height = els.srcCanvas.height;
  octx.clearRect(0,0,els.outCanvas.width, els.outCanvas.height);
  lastOutputBlob = null;
  els.btnDownload.disabled = true;

  setStatus("Image loaded. Click 'Remove Background'.");
  setPerf("");
  // preload model in background (still same page)
  loadModel().catch(err=>{
    console.error(err);
    setStatus("Model load failed. Check console / model path.");
  });
});

els.btnRemove.addEventListener("click", ()=>{
  removeBackground().catch(err=>{
    console.error(err);
    setStatus("Error: " + (err?.message || err));
    els.btnRemove.disabled = false;
  });
});

els.btnDownload.addEventListener("click", downloadPNG);

// Optional demo (small)
els.btnDemo.addEventListener("click", async ()=>{
  // You can replace this demo URL with your own image in repo, e.g. ./demo.jpg
  const demoUrl = "https://images.unsplash.com/photo-1520975916090-3105956dac38?auto=format&fit=crop&w=800&q=80";
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = async ()=>{
    loadedImage = img;
    drawToCanvas(els.srcCanvas, loadedImage);
    setStatus("Demo loaded. Click 'Remove Background'.");
    await loadModel();
  };
  img.src = demoUrl;
});

// Initial
setStatus("Open page → upload image → remove background.");
