const $ = (selector) => document.querySelector(selector);
const sourceCanvas = $("#source");
const outputCanvas = $("#output");
const sourceCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
const outputCtx = outputCanvas.getContext("2d");
const differenceCanvas = $("#difference");
const differenceCtx = differenceCanvas.getContext("2d");
const chartCanvas = $("#benchmarkChart");
const chartCtx = chartCanvas.getContext("2d");
const statusEl = $("#status");

let device = null;
let pipeline = null;
let sourcePixels = null;
let sourceWidth = 512;
let sourceHeight = 512;
let lastCpuResult = null;
let lastGpuResult = null;

function setStatus(message, error = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", error);
}

function makeTestImage(width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const checker = ((Math.floor(x / 32) + Math.floor(y / 32)) % 2) * 55;
      data[i] = (x / width) * 190 + checker;
      data[i + 1] = (y / height) * 190 + 25;
      data[i + 2] = 210 - (x / width) * 120;
      data[i + 3] = 255;
    }
  }
  return new ImageData(data, width, height);
}

function setSource(imageData) {
  lastCpuResult = null;
  lastGpuResult = null;
  differenceCanvas.width = imageData.width;
  differenceCanvas.height = imageData.height;
  differenceCtx.clearRect(0, 0, differenceCanvas.width, differenceCanvas.height);
  sourceWidth = imageData.width;
  sourceHeight = imageData.height;
  sourceCanvas.width = sourceWidth;
  sourceCanvas.height = sourceHeight;
  outputCanvas.width = sourceWidth;
  outputCanvas.height = sourceHeight;
  sourceCtx.putImageData(imageData, 0, 0);
  sourcePixels = new Float32Array(imageData.data.length);
  for (let i = 0; i < imageData.data.length; i++) sourcePixels[i] = imageData.data[i];
  $("#imageInfo").textContent = `Текущий источник: ${sourceWidth} × ${sourceHeight}`;
}

function pixelsToImageData(pixels, width, height) {
  const bytes = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < pixels.length; i++) bytes[i] = Math.max(0, Math.min(255, Math.round(pixels[i])));
  return new ImageData(bytes, width, height);
}

function cpuFilter(input, width, height, algorithm, radius) {
  const output = new Float32Array(input.length);
  const clamp = (v, max) => Math.max(0, Math.min(max - 1, v));
  const lum = (x, y) => {
    const i = (clamp(y, height) * width + clamp(x, width)) * 4;
    return 0.299 * input[i] + 0.587 * input[i + 1] + 0.114 * input[i + 2];
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const out = (y * width + x) * 4;
      if (algorithm === "blur") {
        let r = 0, g = 0, b = 0, count = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const sx = clamp(x + dx, width);
            const sy = clamp(y + dy, height);
            const si = (sy * width + sx) * 4;
            r += input[si]; g += input[si + 1]; b += input[si + 2]; count++;
          }
        }
        output[out] = r / count;
        output[out + 1] = g / count;
        output[out + 2] = b / count;
      } else {
        const tl = lum(x - 1, y - 1), tc = lum(x, y - 1), tr = lum(x + 1, y - 1);
        const ml = lum(x - 1, y), mr = lum(x + 1, y);
        const bl = lum(x - 1, y + 1), bc = lum(x, y + 1), br = lum(x + 1, y + 1);
        const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
        const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
        const edge = Math.min(255, Math.sqrt(gx * gx + gy * gy));
        output[out] = edge;
        output[out + 1] = edge;
        output[out + 2] = edge;
      }
      output[out + 3] = 255;
    }
  }
  return output;
}

async function initializeGPU() {
  if (!navigator.gpu) throw new Error("WebGPU не обнаружен. Нужен совместимый браузер и безопасный контекст (localhost/HTTPS).");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("Браузер не смог получить GPU-адаптер.");
  const newDevice = await adapter.requestDevice();
  device = newDevice;
  newDevice.lost.then((info) => {
    // Не сбрасываем более новое устройство, если старое потерялось позднее.
    if (device === newDevice) {
      device = null;
      pipeline = null;
      setStatus("GPU-устройство потеряно: " + (info.message || info.reason), true);
    }
  });

  const response = await fetch("./shaders/image.wgsl");
  if (!response.ok) throw new Error("Не удалось загрузить shaders/image.wgsl (" + response.status + ").");
  const module = device.createShaderModule({ code: await response.text() });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error("Ошибка WGSL: " + errors.map((e) => e.message).join("; "));
  pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  return true;
}

async function gpuFilter(input, width, height, algorithm, radius) {
  if (!device || !pipeline) await initializeGPU();
  const byteLength = input.byteLength;
  const maxStorageSize = device.limits.maxStorageBufferBindingSize;
  if (byteLength > maxStorageSize) {
    throw new Error(`Изображение слишком большое для этого GPU: требуется ${(byteLength / 1048576).toFixed(1)} MiB, лимит storage buffer — ${(maxStorageSize / 1048576).toFixed(1)} MiB.`);
  }
  const start = performance.now();
  const inputBuffer = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuffer = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuffer = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const paramsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  try {
    device.queue.writeBuffer(inputBuffer, 0, input);
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([width, height, algorithm === "blur" ? 0 : 1, radius]));
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } }
      ]
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, byteLength);

    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    const elapsed = performance.now() - start;
    return { result, elapsed };
  } finally {
    inputBuffer.destroy();
    outputBuffer.destroy();
    readBuffer.destroy();
    paramsBuffer.destroy();
  }
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function comparePixels(a, b) {
  let max = 0, sum = 0;
  for (let i = 0; i < a.length; i++) {
    const difference = Math.abs(a[i] - b[i]);
    max = Math.max(max, difference);
    sum += difference;
  }
  return { max, mean: sum / a.length };
}

function resizePixels(input, fromWidth, fromHeight, size) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = size; tempCanvas.height = size;
  const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
  const image = pixelsToImageData(input, fromWidth, fromHeight);
  const staging = document.createElement("canvas");
  staging.width = fromWidth; staging.height = fromHeight;
  staging.getContext("2d").putImageData(image, 0, 0);
  tempCtx.drawImage(staging, 0, 0, size, size);
  const bytes = tempCtx.getImageData(0, 0, size, size).data;
  const floats = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) floats[i] = bytes[i];
  return floats;
}

function renderDifference(cpu, gpu, width, height) {
  if (!cpu || !gpu || cpu.length !== gpu.length) return;
  const data = new Uint8ClampedArray(cpu.length);
  let maxDifference = 0;
  let totalDifference = 0;
  const pixelCount = cpu.length / 4;
  for (let i = 0; i < cpu.length; i += 4) {
    const difference = Math.max(Math.abs(cpu[i] - gpu[i]), Math.abs(cpu[i + 1] - gpu[i + 1]), Math.abs(cpu[i + 2] - gpu[i + 2]));
    maxDifference = Math.max(maxDifference, difference);
    totalDifference += difference;
    data[i] = Math.min(255, difference * 8);
    data[i + 1] = Math.min(255, difference * 2);
    data[i + 2] = 0;
    data[i + 3] = 255;
  }
  differenceCanvas.width = width;
  differenceCanvas.height = height;
  differenceCtx.putImageData(new ImageData(data, width, height), 0, 0);
  $("#differenceInfo").textContent = `Максимальная разница каналов: ${maxDifference.toFixed(3)}; средняя разница на пиксель: ${(totalDifference / pixelCount).toFixed(4)}. Цвет усилен ×8 для наглядности.`;
}

function updateDifferenceIfReady(width = sourceWidth, height = sourceHeight) {
  if (lastCpuResult && lastGpuResult && lastCpuResult.length === lastGpuResult.length) {
    renderDifference(lastCpuResult, lastGpuResult, width, height);
  }
}

function drawBenchmarkChart(rows) {
  const canvas = chartCanvas;
  const ctx = chartCtx;
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b1423";
  ctx.fillRect(0, 0, width, height);
  const pad = { left: 62, right: 24, top: 24, bottom: 48 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const maxValue = Math.max(1, ...rows.flatMap(row => [row.cpu, row.gpu])) * 1.15;
  ctx.font = "13px system-ui";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let tick = 0; tick <= 4; tick++) {
    const value = maxValue * tick / 4;
    const y = pad.top + plotHeight - plotHeight * tick / 4;
    ctx.strokeStyle = "#293b59";
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillStyle = "#aebfda";
    ctx.fillText(value.toFixed(1), pad.left - 9, y);
  }
  const groupWidth = plotWidth / rows.length;
  const barWidth = Math.min(54, groupWidth * 0.28);
  rows.forEach((row, index) => {
    const center = pad.left + groupWidth * (index + 0.5);
    const cpuHeight = row.cpu / maxValue * plotHeight;
    const gpuHeight = row.gpu / maxValue * plotHeight;
    ctx.fillStyle = "#8bb4ff";
    ctx.fillRect(center - barWidth - 3, pad.top + plotHeight - cpuHeight, barWidth, cpuHeight);
    ctx.fillStyle = "#9fe3c1";
    ctx.fillRect(center + 3, pad.top + plotHeight - gpuHeight, barWidth, gpuHeight);
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = "#dce7f8";
    ctx.fillText(`${row.size}²`, center, height - pad.bottom + 10);
  });
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillStyle = "#8bb4ff"; ctx.fillRect(pad.left, 5, 12, 12);
  ctx.fillStyle = "#dce7f8"; ctx.fillText("CPU", pad.left + 18, 4);
  ctx.fillStyle = "#9fe3c1"; ctx.fillRect(pad.left + 72, 5, 12, 12);
  ctx.fillStyle = "#dce7f8"; ctx.fillText("GPU end-to-end", pad.left + 90, 4);
}

function renderResult(pixels, width, height) {
  outputCanvas.width = width; outputCanvas.height = height;
  outputCtx.putImageData(pixelsToImageData(pixels, width, height), 0, 0);
}

async function runSingle(mode) {
  const algorithm = $("#algorithm").value;
  const radius = Number($("#radius").value);
  const input = sourcePixels;
  const start = performance.now();
  if (mode === "cpu") {
    const result = cpuFilter(input, sourceWidth, sourceHeight, algorithm, radius);
    lastCpuResult = result;
    updateDifferenceIfReady();
    renderResult(result, sourceWidth, sourceHeight);
    setStatus(`CPU: ${(performance.now() - start).toFixed(2)} мс`);
  } else {
    const { result, elapsed } = await gpuFilter(input, sourceWidth, sourceHeight, algorithm, radius);
    lastGpuResult = result;
    updateDifferenceIfReady();
    renderResult(result, sourceWidth, sourceHeight);
    setStatus(`GPU end-to-end: ${elapsed.toFixed(2)} мс (включая ожидание и readback)`);
  }
}

async function runBenchmark() {
  const algorithm = $("#algorithm").value;
  const radius = Number($("#radius").value);
  const repeats = Number($("#repeats").value);
  const sizes = [256, 512, 1024];
  const rows = [];
  $("#benchmark").disabled = true;
  $("#cpu").disabled = true; $("#gpu").disabled = true;
  $("#results").innerHTML = "<p>Выполняются измерения… Не переключайте вкладку до завершения.</p>";
  try {
    for (const size of sizes) {
      setStatus(`Benchmark: ${size} × ${size}…`);
      const input = resizePixels(sourcePixels, sourceWidth, sourceHeight, size);
      cpuFilter(input, size, size, algorithm, radius); // warm-up
      await gpuFilter(input, size, size, algorithm, radius); // warm-up
      const cpuTimes = [], gpuTimes = [];
      let cpuResult, gpuResult;
      for (let i = 0; i < repeats; i++) {
        let t = performance.now();
        cpuResult = cpuFilter(input, size, size, algorithm, radius);
        cpuTimes.push(performance.now() - t);
        const gpu = await gpuFilter(input, size, size, algorithm, radius);
        gpuResult = gpu.result;
        gpuTimes.push(gpu.elapsed);
      }
      const accuracy = comparePixels(cpuResult, gpuResult);
      rows.push({ size, cpu: median(cpuTimes), gpu: median(gpuTimes), accuracy });
    }
    $("#results").innerHTML = `<table><thead><tr><th>Размер</th><th>CPU, мс</th><th>GPU end-to-end, мс</th><th>Отношение CPU/GPU</th></tr></thead><tbody>${rows.map(row => {
      const ratio = row.cpu / row.gpu;
      const label = ratio > 1.05 ? `GPU быстрее в ${ratio.toFixed(2)}×` : ratio < 0.95 ? `CPU быстрее в ${(1 / ratio).toFixed(2)}×` : "примерно одинаково";
      return `<tr><td>${row.size} × ${row.size}</td><td>${row.cpu.toFixed(2)}</td><td>${row.gpu.toFixed(2)}</td><td>${label}</td></tr>`;
    }).join("")}</tbody></table>`;
    drawBenchmarkChart(rows);
    lastCpuResult = cpuFilter(sourcePixels, sourceWidth, sourceHeight, algorithm, radius);
    const currentGpu = await gpuFilter(sourcePixels, sourceWidth, sourceHeight, algorithm, radius);
    lastGpuResult = currentGpu.result;
    updateDifferenceIfReady();
    const worst = rows.reduce((a, b) => b.accuracy.max > a.accuracy.max ? b : a);
    $("#accuracy").textContent = `Проверка корректности (CPU vs GPU, размер ${worst.size}²): максимальная разница ${worst.accuracy.max.toFixed(3)}, средняя абсолютная разница ${worst.accuracy.mean.toFixed(4)} уровня яркости. Небольшие расхождения возможны из-за порядка операций с плавающей точкой.`;
    renderResult(cpuFilter(sourcePixels, sourceWidth, sourceHeight, algorithm, radius), sourceWidth, sourceHeight);
    setStatus("Benchmark завершён. Повторите тест несколько раз для оценки стабильности.");
  } catch (error) {
    setStatus("Ошибка benchmark: " + error.message, true);
    $("#results").textContent = "Не удалось завершить измерения.";
  } finally {
    $("#benchmark").disabled = false;
    $("#cpu").disabled = false; $("#gpu").disabled = false;
  }
}

$("#radius").addEventListener("input", () => $("#radiusValue").textContent = $("#radius").value);
$("#size").addEventListener("change", () => {
  const size = Number($("#size").value);
  setSource(makeTestImage(size, size));
});
$("#file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    if (!file.type.startsWith("image/")) throw new Error("Выберите файл изображения.");
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, width, height);
    setSource(ctx.getImageData(0, 0, width, height));
    $("#size").value = "512";
    setStatus("Изображение загружено. Максимальная сторона ограничена 1024 px.");
  } catch (error) {
    setStatus("Ошибка загрузки изображения: " + error.message, true);
  }
});
$("#cpu").addEventListener("click", () => runSingle("cpu").catch(error => setStatus(error.message, true)));
$("#gpu").addEventListener("click", () => runSingle("gpu").catch(error => setStatus(error.message, true)));
$("#benchmark").addEventListener("click", runBenchmark);

function resetDemo() {
  $("#algorithm").value = "blur";
  $("#radius").value = "2";
  $("#radiusValue").textContent = "2";
  $("#repeats").value = "5";
  $("#size").value = "512";
  $("#file").value = "";
  setSource(makeTestImage(512, 512));
  outputCanvas.width = 512; outputCanvas.height = 512;
  outputCtx.clearRect(0, 0, 512, 512);
  chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
  $("#results").innerHTML = '<p class="muted">Нажмите «Сравнить CPU и GPU», чтобы выполнить измерения.</p>';
  $("#accuracy").textContent = "";
  $("#differenceInfo").textContent = "Сначала запустите CPU и GPU или полный benchmark.";
  setStatus("Настройки сброшены. Загружено тестовое изображение 512 × 512.");
}

async function runDemo() {
  const demoButton = $("#demo");
  demoButton.disabled = true;
  try {
    setStatus("Демонстрация 1/3: выполняется CPU…");
    await runSingle("cpu");
    setStatus("Демонстрация 2/3: выполняется GPU…");
    await runSingle("gpu");
    setStatus("Демонстрация 3/3: запускается benchmark…");
    await runBenchmark();
    setStatus("Демонстрация завершена: результаты, график и разница CPU/GPU готовы.");
  } catch (error) {
    setStatus("Ошибка демонстрации: " + error.message, true);
  } finally {
    demoButton.disabled = false;
  }
}

$("#reset").addEventListener("click", resetDemo);
$("#demo").addEventListener("click", runDemo);
$("#algorithm").addEventListener("change", () => { lastCpuResult = null; lastGpuResult = null; differenceCtx.clearRect(0, 0, differenceCanvas.width, differenceCanvas.height); });
setSource(makeTestImage(512, 512));
$("#differenceInfo").textContent = "Сначала запустите CPU и GPU или полный benchmark.";
initializeGPU().then(() => setStatus("WebGPU готов. Выберите алгоритм и запустите обработку."))
  .catch(error => setStatus(error.message, true));
