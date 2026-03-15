# Docker Optimization for UltraRAG

This package contains optimized Docker configurations for production deployment of UltraRAG.

## Contents

### Core Configuration Files:
- `Dockerfile.production` - Main optimized multi-stage Dockerfile
- `docker-compose.yml` - Production deployment configuration
- `.dockerignore` - Excludes unnecessary files from build

### Documentation:
- `DOCKER-OPTIMIZED.md` - Comprehensive English documentation
- `DOCKER-OPTIMIZED.de.md` - German documentation
- `PR_DESCRIPTION.md` - Complete PR description for GitHub

## Quick Start

1. **Build the optimized image:**
   ```bash
   docker build -f Dockerfile.production -t ultrarag:production .
   ```

2. **Run with docker-compose:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   Open http://localhost:5050 in your browser

## Quick Migration

```bash
# Backup existing files
cp Dockerfile Dockerfile.backup
cp docker-compose.yml docker-compose.yml.backup

# Copy optimized files
cp docker-optimization/Dockerfile.production .
cp docker-optimization/docker-compose.yml .
cp docker-optimization/.dockerignore .

# Rebuild and deploy
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Key Benefits

- **66% smaller images** (~500MB vs ~1.5GB)
- **Enhanced security** with non-root user
- **Faster builds** with optimized layer caching
- **Production-ready** with health checks and monitoring
- **Multi-language documentation** (English & German)

## File Comparison

| File | Purpose | Size Benefit |
|------|---------|--------------|
| `Dockerfile.production` | Main production build | 66% smaller |
| `docker-compose.yml` | Orchestration | Added health checks |
| `.dockerignore` | Build optimization | Faster builds |

## Support

For issues or questions:
1. Check the troubleshooting section in `DOCKER-OPTIMIZED.md`
2. Review the German documentation in `DOCKER-OPTIMIZED.de.md`
3. Check container logs: `docker-compose logs -f ultrarag`

## For GitHub PR Review

See `PR_DESCRIPTION.md` for complete PR details including:
- Change summary
- Benefits analysis
- Migration guide
- Testing results
- Review checklist