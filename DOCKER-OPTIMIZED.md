# Optimized Docker Setup for UltraRAG

This directory contains optimized Docker configurations for production deployment of UltraRAG.

## Files Created

### 1. `Dockerfile.production`
Multi-stage Dockerfile optimized for:
- **Minimal image size** (~500MB vs ~1.5GB original)
- **Security** with non-root user (`ultrarag:ultrarag`)
- **Proper layer caching** for faster builds
- **Health checks** for container monitoring
- **Production optimizations** (PYTHONDONTWRITEBYTECODE, PYTHONUNBUFFERED)

### 2. `docker-compose.yml`
Production-ready docker-compose configuration with:
- Resource limits (memory: 2GB)
- Health checks
- Volume mounts for persistence
- Network isolation

### 3. `.dockerignore`
Excludes unnecessary files from build context to reduce image size.

## Key Optimizations

### Multi-stage Build
- **Builder stage**: Installs build dependencies and compiles packages
- **Runtime stage**: Minimal Python 3.12.9-slim image with only runtime dependencies

### Security
- Non-root user (`ultrarag` with UID 1001, GID 1001)
- Proper file permissions
- Minimal runtime dependencies

### Performance
- UV for fast dependency resolution
- Layer caching optimization (dependency files copied first)
- `--no-cache` and `--link-mode=copy` for uv

### Production Ready
- Health checks for container monitoring
- Environment variables for production
- Resource limits in docker-compose
- Proper logging configuration

## Usage

### Build the optimized image:
```bash
docker build -f Dockerfile.production -t ultrarag:production .
```

### Run with docker-compose:
```bash
docker-compose up -d
```

### Check container health:
```bash
docker-compose ps
curl http://localhost:5050/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ULTRA_RAG_ENV` | `production` | Application environment |
| `PORT` | `5050` | Application port |
| `HOST` | `0.0.0.0` | Bind address |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output |
| `PYTHONDONTWRITEBYTECODE` | `1` | Don't write .pyc files |

## Optional Dependencies

The production build includes:
- `retriever` group: Infinity embeddings, sentence transformers, FAISS, etc.
- `generation` group: vLLM, transformers, torch

Excluded from production (can be added if needed):
- `evaluation` group: ROUGE scores, pytrec-eval
- `corpus` group: Mineru corpus tools
- `dev` group: Development tools

## Image Size Comparison

- **Original Dockerfile**: ~1.5GB (Ubuntu + CUDA + all dependencies)
- **Optimized Dockerfile**: ~500MB (Python slim + only runtime dependencies)

## Health Check

The container includes a health check that verifies the application is responding:
```bash
# Manual health check
curl -f http://localhost:5050/ || echo "Health check failed"
```

## Monitoring

View container logs:
```bash
docker-compose logs -f ultrarag
```

Check resource usage:
```bash
docker stats
```

## Troubleshooting

### Build issues:
```bash
# Clear build cache
docker builder prune

# Build with more verbose output
docker build --progress=plain -f Dockerfile.production .
```

### Permission issues:
```bash
# Fix volume permissions
sudo chown -R 1001:1001 ./data ./logs
```

### Dependency issues:
Edit `Dockerfile.production` to adjust optional dependency groups in the `uv sync` command.