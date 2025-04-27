FROM ollama/ollama:latest

# Create a user and set up permissions
RUN useradd -ms /bin/bash ollama-user

# Set the home directory
ENV HOME=/home/ollama-user
WORKDIR $HOME

# Ensure the user has access to the directory
RUN mkdir -p $HOME/.ollama && chown -R ollama-user:ollama-user $HOME/.ollama

# Install netcat (nc) for checking server readiness
RUN apt-get update && apt-get install -y netcat

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
