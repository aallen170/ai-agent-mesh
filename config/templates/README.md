# config/templates/

Per-tier device config templates for AIMESH worker nodes. Each template maps
to a hardware tier and is used as the starting point for onboarding a specific
device (AIMESH-19 through AIMESH-25).

## Templates

| File | Tier | Devices | Inference runtime |
|------|------|---------|-------------------|
| `tier0_mobile.yaml` | 0 | iPads, iPhone, Android tablet | mlx_lm / MLC-LLM |
| `tier1_igpu_laptop.yaml` | 1 | General laptop (iGPU) | Ollama |
| `tier2_dgpu_desktop.yaml` | 2 | Desktop PC (also hosts control plane) | Ollama |
| `tier2_dgpu_laptop.yaml` | 2 | Gaming laptop | Ollama |

## How to onboard a new device

1. **Copy** the matching template to `config/` and rename it for your device:
   ```
   cp config/templates/tier2_dgpu_laptop.yaml config/gaming_laptop.yaml
   ```

2. **Fill in** every line marked `# TODO` — at minimum:
   - `device_id` — unique slug (e.g. `gaming-laptop`)
   - `name` — display name for Grafana / registry
   - `redis_url` — point at the desktop PC's Redis (e.g. `redis://192.168.1.10:6379/0`)
   - `litellm_url` — point at the desktop PC's LiteLLM (e.g. `http://192.168.1.10:4000/v1`)
   - `models` — uncomment / set the LiteLLM model name(s) for this device
   - `hardware.*` — actual RAM, VRAM, GPU, core count, OS

3. **Start the inference server** on the device (see the template's Prerequisites
   section for the exact commands per tier).

4. **Update `infra/litellm_config.yaml`** if the device's hostname or IP differs
   from the placeholder — the `api_base` for the device's model entry must be
   reachable from the desktop PC.

5. **Start the worker**:
   ```
   python scripts/run_worker.py --config config/gaming_laptop.yaml
   ```
   The worker will register itself with the control-plane registry and begin
   pulling tasks from its tier queue.

## Environment variable overrides

Sensitive or environment-specific values can be passed as env vars instead of
hard-coding them in the YAML file:

| Env var | Overrides |
|---------|-----------|
| `REDIS_URL` | `redis_url` |
| `LITELLM_BASE_URL` | `litellm_url` |
| `LITELLM_MASTER_KEY` | auth key sent to LiteLLM (default: `sk-aimesh-local`) |

## Device config files

Config files for actual devices (`config/gaming_laptop.yaml`, etc.) are
excluded from version control via `.gitignore` (`device_config.yaml` pattern).
Commit only the templates in this directory, not the filled-in device files.
