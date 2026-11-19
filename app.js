const statusEl = document.querySelector("#status");
const canvas = document.querySelector("#plot");
const ctx = canvas.getContext("2d");

let device = null;
let pipeline = null;

async function initializeWebGPU() {
  if (!navigator.gpu) {
    statusEl.textContent = "WebGPU недоступен. Откройте страницу в актуальном браузере с поддержкой WebGPU через localhost или HTTPS.";
    statusEl.style.color = "#ffb4a8";
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("GPU-адаптер не найден");

    device = await adapter.requestDevice();
    const shaderResponse = await fetch("./shaders/compute.wgsl");
    if (!shaderResponse.ok) throw new Error("Не удалось загрузить shaders/compute.wgsl");
    const shaderCode = await shaderResponse.text();

    const shaderModule = device.createShaderModule({ code: shaderCode });
    pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "main" }
    });

    statusEl.textContent = "WebGPU готов к работе — устройство успешно создано.";
    return true;
  } catch (error) {
    statusEl.textContent = "Ошибка инициализации: " + error.message;
    statusEl.style.color = "#ffb4a8";
    return false;
  }
}

function draw(values) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#263656";
  ctx.lineWidth = 1;

  for (let j = 0; j <= 4; j++) {
    const y = j * canvas.height / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#80a5ff";
  ctx.lineWidth = 2;
  ctx.beginPath();

  for (let i = 0; i < values.length; i++) {
    const x = i / (values.length - 1) * canvas.width;
    const y = canvas.height / 2 - values[i] * (canvas.height * 0.42);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

async function runCompute() {
  if (!device && !(await initializeWebGPU())) return;

  const count = Number(document.querySelector("#count").value);
  const outputBuffer = device.createBuffer({
    size: count * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const readBuffer = device.createBuffer({
    size: count * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: outputBuffer } }]
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 256));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, count * 4);

  const start = performance.now();
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const elapsed = performance.now() - start;

  const values = new Float32Array(readBuffer.getMappedRange().slice(0));
  draw(values);

  document.querySelector("#stats").textContent =
    `Вычислено элементов: ${count.toLocaleString("ru-RU")}. Время до готовности результата (включая чтение): ${elapsed.toFixed(2)} мс. Первое выполнение может включать дополнительные затраты на подготовку GPU.`;

  readBuffer.unmap();
  outputBuffer.destroy();
  readBuffer.destroy();
}

document.querySelector("#run").addEventListener("click", runCompute);
initializeWebGPU();