#!/bin/bash
set -e

echo "Building optimized UltraRAG Docker image..."

# Build the production image
docker build \
    -f Dockerfile.production \
    -t ultrarag:production \
    --target runtime \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    .

echo ""
echo "Build complete!"
echo ""
echo "To run the container:"
echo "  docker run -p 5050:5050 ultrarag:production"
echo ""
echo "Or using docker-compose:"
echo "  docker-compose up -d"
echo ""
echo "Image size comparison:"
echo "  Original image: ~1.5GB (estimated)"
echo "  Optimized image: ~500MB (estimated)"
echo ""
echo "To check image size:"
echo "  docker images ultrarag"