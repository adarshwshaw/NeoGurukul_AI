#!/bin/bash

# Start Ollama server in the background
ollama serve &

# Wait for the server to be ready
while ! nc -z localhost 7860; do
    echo "Waiting for Ollama server to start..."
    sleep 1
done

# Pull the model
echo "Pulling the model..."
ollama pull llama3.2

# Keep the container running
wait
