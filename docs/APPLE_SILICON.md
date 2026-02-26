# Apple Silicon (M1/M2/M3) Setup

The Rugby League and AFL training pipelines are configured for Apple Silicon Macs, including M2 Pro with 16GB RAM.

## GPU Acceleration

Install `tensorflow-metal` for Metal GPU acceleration (recommended on M1/M2/M3):

```bash
pip install tensorflow-metal
```

Install after TensorFlow. Compatible with TensorFlow 2.14+.

## Requirements

- macOS 12.0 or later
- Python 3.9–3.11 (3.11 recommended for best compatibility)
- TensorFlow 2.14+

## 16GB RAM

Default batch sizes (16–64) are safe. If you encounter out-of-memory errors during training or Hyperband, reduce batch size in the UI or via CLI (e.g. `--batch-size 16` where supported).

## Configuration

The training scripts automatically:

- Enable memory growth on GPU devices to avoid allocating all VRAM upfront
- Limit thread parallelism (4 intra-op, 4 inter-op) to avoid oversubscription on M2’s 8–12 cores
