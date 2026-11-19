# WebGPU Compute Demo

Небольшое учебное веб-приложение, демонстрирующее вычисления на GPU с помощью WebGPU и WGSL.

## Возможности

- Запрос GPU-адаптера и устройства через WebGPU.
- Загрузка compute shader из отдельного файла `shaders/compute.wgsl`.
- Параллельное вычисление функции `f(x) = sin(x) * cos(3x)`.
- Выбор размера массива: 1 024, 16 384, 65 536 или 262 144 элемента.
- Визуализация результата на Canvas и отображение времени получения результата.

## Структура проекта

```text
webgpu-demo/
├── index.html
├── app.js
├── shaders/
│   └── compute.wgsl
├── README.md
├── .gitignore
└── LICENSE
```

## Требования

- Современный браузер с поддержкой WebGPU.
- Совместимые операционная система, графический процессор и драйвер.
- Python 3 или другой локальный HTTP-сервер.

## Запуск

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/Weaknessknight/webgpu-demo.git
   cd webgpu-demo
   ```

2. Запустите локальный сервер:

   ```bash
   python -m http.server 8000
   ```

3. Откройте в браузере:

   ```text
   http://localhost:8000
   ```

4. Нажмите «Выполнить вычисление» и измените количество точек.

WebGPU требует безопасного контекста. Для локальной разработки `localhost` подходит; для размещённого сайта используйте HTTPS.

## Как работает демо

1. JavaScript получает `GPUAdapter` и `GPUDevice`.
2. Загружает WGSL-код и создаёт compute pipeline.
3. Создаёт storage buffer и readback buffer.
4. Запускает compute shader через `dispatchWorkgroups`.
5. Копирует результаты в буфер, доступный CPU, и отображает график.

Каждый shader invocation обрабатывает один индекс массива. Размер workgroup равен 256; число workgroups округляется вверх, а проверка границ предотвращает выход за пределы массива.

## Ограничения измерения времени

Отображаемое время включает отправку GPU-команд, ожидание завершения и чтение результата обратно в CPU. Это не чистое время исполнения арифметики внутри шейдера. На результат влияют браузер, драйвер, GPU, компиляция и передача данных.

## Если WebGPU недоступен

- Проверьте актуальную поддержку WebGPU для браузера и операционной системы.
- Обновите браузер и графические драйверы.
- Убедитесь, что открываете страницу через `localhost` или HTTPS, а не через `file://`.
- Попробуйте другое совместимое устройство.

## Полезные ссылки

- [W3C WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [MDN: WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
- [W3C WGSL Specification](https://www.w3.org/TR/WGSL/)
- [Chrome Developers: WebGPU](https://developer.chrome.com/docs/web-platform/webgpu/overview)
