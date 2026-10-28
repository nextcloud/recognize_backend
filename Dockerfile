# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/nc_app_recognize_backend_data/huggingface
ENV TRANSFORMERS_CACHE=/nc_app_recognize_backend_data/huggingface
ENV INSIGHTFACE_HOME=/nc_app_recognize_backend_data/insightface

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ffmpeg libsndfile1 \
        libgl1 libglib2.0-0 \
        curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Download and install FRP client with checksum verification
# FRP version and checksums - update these when upgrading
ARG FRP_VERSION=0.71.1
ARG FRP_AMD64_SHA256=1d9875210972a5d1a62b2dc4c48df45ab0af05d3ca1b5e494a014428259df803
ARG FRP_ARM64_SHA256=8edf590b0c396fa42f9c147121a76af83cfc2002b08f5aef0ea1008c43482019

RUN set -ex; \
    ARCH=$(uname -m); \
    if [ "$ARCH" = "aarch64" ]; then \
        FRP_ARCH="arm64"; \
        FRP_SHA256="${FRP_ARM64_SHA256}"; \
    else \
        FRP_ARCH="amd64"; \
        FRP_SHA256="${FRP_AMD64_SHA256}"; \
    fi; \
    FRP_URL="https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/frp_${FRP_VERSION}_linux_${FRP_ARCH}.tar.gz"; \
    echo "Downloading FRP v${FRP_VERSION} for ${FRP_ARCH}..."; \
    curl -fsSL "${FRP_URL}" -o /tmp/frp.tar.gz; \
    ACTUAL_SHA256=$(sha256sum /tmp/frp.tar.gz | cut -d' ' -f1); \
    if [ "$ACTUAL_SHA256" != "$FRP_SHA256" ]; then \
        echo "Checksum verification failed for FRP v${FRP_VERSION} (${FRP_ARCH})"; \
        echo "Expected: ${FRP_SHA256}"; \
        echo "Got:      ${ACTUAL_SHA256}"; \
        exit 1; \
    fi; \
    tar -C /tmp -xzf /tmp/frp.tar.gz; \
    cp /tmp/frp_${FRP_VERSION}_linux_${FRP_ARCH}/frpc /usr/local/bin/frpc; \
    chmod +x /usr/local/bin/frpc; \
    rm -rf /tmp/frp_${FRP_VERSION}_linux_${FRP_ARCH} /tmp/frp.tar.gz; \
    echo "FRP client installed successfully"

COPY requirements.txt /
COPY healthcheck.sh /
COPY --chmod=775 start.sh /

ADD li[b] /app/lib

# insightface ships an sdist only and compiles a Cython extension, so a C/C++
# toolchain is needed at install time. The runtime base image has none; install
# it for the pip step and drop it again in the same layer.
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential; \
    python3 -m pip install --upgrade pip; \
    python3 -m pip install -r /requirements.txt; \
    apt-get purge -y --auto-remove build-essential; \
    rm -rf /var/lib/apt/lists/* /root/.cache /requirements.txt

WORKDIR /app/lib
ENTRYPOINT ["/start.sh", "python3", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/recognize_backend
HEALTHCHECK --interval=2s --timeout=2s --retries=300 CMD /healthcheck.sh
