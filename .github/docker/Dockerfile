# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Install Java (OpenJDK) and sudo
RUN apt-get update && apt-get install -y default-jre sudo git && apt-get clean;

# Automatically detect JAVA_HOME
RUN JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::") && \
    echo "JAVA_HOME=$JAVA_HOME" >> /etc/environment && \
    echo "export JAVA_HOME" >> /etc/profile

# Installing poetry
ENV POETRY_HOME=/opt/poetry
RUN python3 -m venv $POETRY_HOME && \
    . $POETRY_HOME/bin/activate && \
    pip install poetry
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY pyproject.toml ./

RUN poetry install --only main,test --no-root

COPY . ./

RUN poetry install --only-root

CMD ["poetry", "run", "pytest"]