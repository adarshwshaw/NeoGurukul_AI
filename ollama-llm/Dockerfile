# Use NVIDIA CUDA runtime base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04



# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    gnupg \
    netcat \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Create a user and set up permissions
RUN useradd -ms /bin/bash ollama-user

# Set the home directory
ENV HOME=/home/ollama-user
WORKDIR $HOME

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the entrypoint script before switching users
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# Set permissions for the entrypoint script
RUN chmod +x /usr/local/bin/entrypoint.sh

# Switch to the non-root user
USER ollama-user

# Set Ollama to listen on all network interfaces
ENV OLLAMA_HOST=0.0.0.0:7860

# Expose the default port
EXPOSE 7860

# Use the custom entrypoint script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
