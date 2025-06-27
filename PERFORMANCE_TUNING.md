# Performance Tuning (M2 8GB)

## 1. Memory Management
- **Target**: <1GB RSS
- **Tools**:
  ```bash
  pip install guppy3
  ```
  ```python
  from guppy import hpy
  h = hpy()
  print(h.heap())
  ```

## 2. Latency Targets
- Feature Extraction: <20ms
- SER Model: <30ms
- TTS: <60ms
- Vocoder: <30ms
- **Total**: <120ms

## 3. MPS Optimization
```python
# Use MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Mixed precision
with torch.amp.autocast(device_type="mps", dtype=torch.float16):
    output = model(input)
```

## 4. Model Quantization
```python
# Dynamic quantization for SerModel
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## 5. Memory Cleanup
```python
import gc
import torch

def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
```

## 6. Monitoring
```bash
# CPU/Memory usage
top -o mem

# GPU usage (MPS)
sudo powermetrics --samplers gpu_power -i 1000
```

## 7. Common Issues
1. **High Memory Usage**
   - Reduce batch size
   - Use smaller models
   - Clear cache regularly

2. **High Latency**
   - Profile with `time.perf_counter()`
   - Consider Core ML conversion
   - Optimize data loading

3. **MPS Errors**
   - Update PyTorch to latest nightly
   - Check macOS version (12.3+)
   - Verify M1/M2 chip support
