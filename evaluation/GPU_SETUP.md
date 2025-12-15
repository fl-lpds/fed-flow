# GPU Setup for Docker

This guide explains how to set up GPU support for Docker containers in the federated learning system.

## Prerequisites

1. **NVIDIA GPU** with compatible drivers installed
2. **NVIDIA Container Toolkit** installed
3. **Docker** version 19.03 or later
4. **Docker Compose** version 1.28.0 or later (for `deploy.resources` syntax)

## Installation Steps

### 1. Install NVIDIA Container Toolkit

#### Ubuntu/Debian:
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

#### CentOS/RHEL:
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

# Install nvidia-container-toolkit
sudo yum install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### 2. Verify Installation

```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

If this command works, GPU support is properly configured.

## Using GPU with Docker Compose

The generated `docker-compose.yml` files use two methods for GPU access:

### Method 1: `runtime: nvidia` (Compatible)
This is the older, more compatible method that works with `nvidia-docker2`:
```yaml
runtime: nvidia
environment:
  NVIDIA_VISIBLE_DEVICES: "0,1"  # Optional: specify GPU IDs
```

### Method 2: `deploy.resources.reservations.devices` (Modern)
This is the newer method that requires `nvidia-container-toolkit`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # or device_ids: ['0', '1']
          capabilities: [gpu]
```

## Troubleshooting

### Error: "could not select device driver 'nvidia'"

This error means the NVIDIA Container Toolkit is not installed or Docker is not configured to use it.

**Solution:**
1. Install NVIDIA Container Toolkit (see Installation Steps above)
2. Verify Docker can see NVIDIA runtime:
   ```bash
   docker info | grep -i runtime
   ```
   You should see `nvidia` in the list of runtimes.

3. If `nvidia` runtime is not listed, configure Docker daemon:
   ```bash
   sudo mkdir -p /etc/docker
   sudo tee /etc/docker/daemon.json <<EOF
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     },
     "default-runtime": "runc"
   }
   EOF
   sudo systemctl restart docker
   ```

### Fallback: Use CPU Only

If GPU setup is not possible, you can disable GPU support:

1. When creating the scenario, answer "No" to "Enable GPU support?"
2. Or manually edit `docker-compose.yml` and remove:
   - `runtime: nvidia` lines
   - `deploy.resources.reservations.devices` sections
   - GPU-related environment variables

The code will automatically fall back to CPU usage.

## Environment Variables

The following environment variables control GPU usage:

- `CUDA_VISIBLE_DEVICES`: Controls which GPUs are visible to CUDA (e.g., "0,1")
- `NVIDIA_VISIBLE_DEVICES`: Controls which GPUs are visible to NVIDIA runtime (e.g., "0,1")
- `GPU_DEVICE_ID`: Specifies which GPU device ID to use in PyTorch (e.g., "0")
- `FORCE_CPU`: Force CPU usage even if GPU is available (set to "True")

## Testing GPU Access

After starting containers, verify GPU access:

```bash
# Check if container can see GPU
docker exec <container_name> nvidia-smi

# Or check from inside the container
docker exec <container_name> python3 -c "import torch; print(torch.cuda.is_available())"
```

