# docker buildx build -t ml-2025 .
# docker run -it ml-2025

# Noble Numbat, 24.04 LTS
FROM ubuntu:noble AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        sudo \
        vim && \
    apt-get clean

WORKDIR /app
COPY . .

RUN useradd --create-home ml && \
    chown -R ml:ml /app && \
    usermod -aG sudo ml && \
    echo "ml ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ml
ENTRYPOINT ["bash"]
