FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get --yes --no-install-recommends install python3-dev build-essential cmake curl && rm -rf /var/lib/apt/lists/*
EXPOSE 18080
ENTRYPOINT [ "bash", "./docker_startup.sh" ]

