# Benchmark Inference — PPE Detection

- CPU: `Intel64 Family 6 Model 154 Stepping 4, GenuineIntel`
- Resolusi model: `(416, 416)`
- Frame diuji: 10 gambar, 50 iterasi (+5 warmup)
- Latency = end-to-end per frame (preprocess + inference + NMS).

| Backend | Device | Mean (ms) | Median (ms) | p95 (ms) | FPS | Speedup |
|---------|--------|-----------|-------------|----------|-----|---------|
| torch | CPU | 155.8 | 146.2 | 233.7 | 6.4 | 1.00x |
| openvino | CPU | 84.4 | 79.0 | 135.1 | 11.8 | 1.85x |
| openvino | GPU | 22.9 | 21.4 | 33.3 | 43.7 | 6.81x |
| openvino-int8 | CPU | 70.0 | 59.8 | 149.3 | 14.3 | 2.22x |
| openvino-int8 | GPU | 18.2 | 18.1 | 22.2 | 54.9 | 8.55x |

> Speedup dihitung relatif terhadap backend `torch` (PyTorch CPU).
