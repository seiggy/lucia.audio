# Lucia.Audio

Wyoming protocol compatible TTS service powered by [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) from Resemble AI. Designed for low-latency voice synthesis on NVIDIA GPUs with a built-in web dashboard for voice profile management.

## Features

- **Wyoming Protocol** — Drop-in TTS service for Home Assistant voice pipelines (port 10200)
- **Chatterbox Turbo** — 350M parameter model with zero-shot voice cloning, paralinguistic tags (`[laugh]`, `[cough]`, etc.)
- **Voice Dashboard** — Web UI (port 8095) for uploading audio clips and managing voice profiles
- **GPU Optimized** — FP16 inference, pre-computed voice conditionals, model warm-up for minimum latency
- **Docker Ready** — Single container with NVIDIA GPU support

## Quick Start

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA GPU (tested on RTX 3080)

### Run with Docker Compose

```bash
git clone https://github.com/your-user/lucia.audio.git
cd lucia.audio
docker compose up -d
```

The first startup will download the Chatterbox Turbo model weights (~1.5GB). Subsequent starts will use the cached weights.

### Services

| Service | Port | Description |
|---------|------|-------------|
| Wyoming TTS | 10200 | Wyoming protocol TCP server |
| Dashboard | 8095 | Voice management web UI |

## Voice Profiles

1. Open the dashboard at `http://your-server:8095`
2. Upload a WAV audio clip (5+ seconds of clear speech)
3. Enter a name for the voice profile
4. Click "Upload & Process Voice" — conditionals are pre-computed for fast inference
5. Use the TTS test section to try it out

Voice profiles are stored in `/data/voices` and persist across container restarts.

## Home Assistant Integration

1. Go to **Settings → Integrations → Add Integration**
2. Search for **Wyoming Protocol**
3. Enter host and port: `your-server:10200`
4. Select a voice from your uploaded profiles in the Assist pipeline settings

## Configuration

Override defaults via the `command` key in `docker-compose.yml` or pass arguments directly:

```bash
docker run --gpus all -p 10200:10200 -p 8095:8095 -v lucia-data:/data lucia-tts \
    --uri tcp://0.0.0.0:10200 \
    --voices-dir /data/voices \
    --dashboard-port 8095 \
    --default-voice "Lucia" \
    --zeroconf
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--uri` | `tcp://0.0.0.0:10200` | Wyoming server URI |
| `--voices-dir` | `/data/voices` | Voice profiles storage path |
| `--dashboard-port` | `8095` | Web dashboard port (0 to disable) |
| `--default-voice` | `None` | Default voice profile name |
| `--device` | `cuda` | Torch device (`cuda` or `cpu`) |
| `--no-half` | `false` | Disable FP16 inference |
| `--zeroconf` | disabled | Enable Home Assistant auto-discovery |
| `--debug` | `false` | Enable debug logging |

## Architecture

```
┌─────────────────────────────────────────┐
│            Docker Container             │
│                                         │
│  Wyoming TTS Server (:10200)            │
│  ├── Handles Synthesize events          │
│  ├── Resolves voice by name → profile   │
│  └── Returns PCM audio chunks           │
│                                         │
│  Dashboard (:8095)                      │
│  ├── Upload/manage voice profiles       │
│  └── Test TTS synthesis                 │
│                                         │
│  TTS Engine (ChatterboxTurboTTS)        │
│  ├── FP16 inference on GPU              │
│  ├── Pre-computed voice conditionals    │
│  └── Thread-safe async inference        │
│                                         │
│  /data/voices/                          │
│  ├── {id}/reference.wav                 │
│  ├── {id}/conds.pt (cached)             │
│  └── {id}/profile.json                  │
└─────────────────────────────────────────┘
```

## Performance Notes

- Voice conditionals are pre-computed on upload and cached as `.pt` files, avoiding re-encoding on every TTS request
- Model runs in FP16 by default on CUDA for ~2x memory savings and faster inference
- A warm-up synthesis runs at startup to ensure CUDA kernels are compiled
- The Wyoming handler uses asyncio with a thread pool executor for non-blocking GPU inference

## Development

```bash
# Install locally (requires Python 3.11+)
pip install -e .

# Run directly
python -m wyoming_chatterbox \
    --uri tcp://0.0.0.0:10200 \
    --voices-dir ./voices \
    --device cuda
```

## License

MIT — See [LICENSE](LICENSE)

## Credits

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI
- [Wyoming Protocol](https://github.com/OHF-Voice/wyoming) by OHF Voice / Rhasspy
