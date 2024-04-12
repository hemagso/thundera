# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Install Java (OpenJDK) and sudo
RUN apt-get update && apt-get install -y default-jre sudo git && apt-get clean;

# Automatically detect JAVA_HOME
RUN JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::") && \
    echo "JAVA_HOME=$JAVA_HOME" >> /etc/environment && \
    echo "export JAVA_HOME" >> /etc/profile

# Install PySpark
RUN pip install pyspark

# Create a non-root user called 'developer' and give them sudo privileges
RUN useradd -m developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer

# Set the working directory
WORKDIR /workspace

# Change to non-root privilege
USER developer
