
# make docker-build docker-run

# Noble Numbat, 24.04 LTS
FROM ubuntu:noble AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        curl \
        file \
        g++ \
        git \
        net-tools \
        pkg-config \
        python-is-python3 \
        python3-pip \
        python3-venv \
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
RUN (curl -LsSf https://astral.sh/uv/install.sh | sh) && \
    source ~/.bashrc && \
    time make install

ENTRYPOINT ["bash"]
