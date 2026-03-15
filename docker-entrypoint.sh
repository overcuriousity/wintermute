#!/bin/bash
set -e

# Function to wait for dependencies
wait_for_dependencies() {
    if [ -n "$WAIT_FOR_HOSTS" ]; then
        echo "Waiting for dependencies..."
        for host_port in $(echo $WAIT_FOR_HOSTS | tr ',' ' '); do
            host=$(echo $host_port | cut -d: -f1)
            port=$(echo $host_port | cut -d: -f2)
            echo "Waiting for $host:$port..."
            while ! nc -z $host $port; do
                sleep 1
            done
            echo "$host:$port is available"
        done
    fi
}

# Function to check if we need to run database migrations
run_migrations() {
    if [ "$RUN_MIGRATIONS" = "true" ]; then
        echo "Running database migrations..."
        # Add migration commands here if needed
        # python -m alembic upgrade head
    fi
}

# Function to initialize the application
initialize_app() {
    echo "Initializing UltraRAG..."
    
    # Set default environment variables if not set
    export PORT=${PORT:-5050}
    export HOST=${HOST:-0.0.0.0}
    
    # Create necessary directories
    mkdir -p /app/data/logs
    mkdir -p /app/data/cache
    
    # Set proper permissions
    chown -R ultrarag:ultrarag /app/data 2>/dev/null || true
}

# Main execution
main() {
    echo "Starting UltraRAG container..."
    
    # Wait for dependencies if specified
    wait_for_dependencies
    
    # Initialize application
    initialize_app
    
    # Run migrations if needed
    run_migrations
    
    # Execute the command
    echo "Starting UltraRAG with command: $@"
    exec "$@"
}

# Run main function with all arguments
main "$@"