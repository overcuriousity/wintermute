#!/bin/bash
set -e

echo "========================================="
echo "UltraRAG Docker Migration Script"
echo "========================================="
echo ""
echo "This script helps migrate from the original Docker setup"
echo "to the optimized production configuration."
echo ""

# Check if running in UltraRAG directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Not in UltraRAG directory"
    echo "Please run this script from the UltraRAG project root"
    exit 1
fi

echo "📋 Current directory: $(pwd)"
echo ""

# Backup existing files
echo "🔒 Creating backups..."
BACKUP_DIR="docker-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "Dockerfile" ]; then
    cp Dockerfile "$BACKUP_DIR/Dockerfile"
    echo "  ✓ Backed up Dockerfile"
fi

if [ -f "docker-compose.yml" ]; then
    cp docker-compose.yml "$BACKUP_DIR/docker-compose.yml"
    echo "  ✓ Backed up docker-compose.yml"
fi

if [ -f ".dockerignore" ]; then
    cp .dockerignore "$BACKUP_DIR/.dockerignore"
    echo "  ✓ Backed up .dockerignore"
fi

echo ""
echo "📦 Copying optimized files..."

# Copy optimized files
cp docker-optimization/Dockerfile.production .
cp docker-optimization/docker-compose.yml .
cp docker-optimization/.dockerignore .

echo "  ✓ Copied Dockerfile.production"
echo "  ✓ Copied docker-compose.yml"
echo "  ✓ Copied .dockerignore"

echo ""
echo "📚 Documentation available:"
echo "  - docker-optimization/DOCKER-OPTIMIZED.md (English)"
echo "  - docker-optimization/DOCKER-OPTIMIZED.de.md (German)"
echo "  - docker-optimization/COMPARISON.md (Comparison)"
echo "  - docker-optimization/PR_DESCRIPTION.md (PR Details)"

echo ""
echo "🚀 Migration complete!"
echo ""
echo "Next steps:"
echo "1. Review the new configuration:"
echo "   - cat Dockerfile.production"
echo "   - cat docker-compose.yml"
echo ""
echo "2. Test the optimized build:"
echo "   - docker build -f Dockerfile.production -t ultrarag:production ."
echo ""
echo "3. Deploy with docker-compose:"
echo "   - docker-compose down"
echo "   - docker-compose build --no-cache"
echo "   - docker-compose up -d"
echo ""
echo "4. Verify the deployment:"
echo "   - docker-compose ps"
echo "   - curl http://localhost:5050/"
echo ""
echo "📁 Backup files saved in: $BACKUP_DIR"
echo "You can restore original files from this directory if needed."
echo ""
echo "For more details, see docker-optimization/README.md"