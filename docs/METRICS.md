# Metrics Akurasi — PPE Detection

Split: **test** · imgsz **416** · dataset `ppe-detection-2`

| Varian | mAP@50 | mAP@50-95 | Precision | Recall | Δ mAP@50 |
|--------|--------|-----------|-----------|--------|----------|
| PyTorch FP32 | 0.6545 | 0.3886 | 0.6756 | 0.6282 | — |
| OpenVINO FP32 | 0.6547 | 0.3845 | 0.6682 | 0.6277 | +0.02 pp |
| OpenVINO INT8 | 0.6513 | 0.3820 | 0.6636 | 0.6298 | -0.32 pp |

> Δ dihitung terhadap varian pertama (PyTorch FP32).
