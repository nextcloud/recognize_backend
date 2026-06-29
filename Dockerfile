# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

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

# FRP client for AppAPI/HaRP tunnels
RUN set -ex; \
    ARCH=$(uname -m); \
    if [ "$ARCH" = "aarch64" ]; then \
      FRP_URL="https://raw.githubusercontent.com/nextcloud/HaRP/main/exapps_dev/frp_0.61.1_linux_arm64.tar.gz"; \
    else \
      FRP_URL="https://raw.githubusercontent.com/nextcloud/HaRP/main/exapps_dev/frp_0.61.1_linux_amd64.tar.gz"; \
    fi; \
    curl -L "$FRP_URL" -o /tmp/frp.tar.gz; \
    tar -C /tmp -xzf /tmp/frp.tar.gz; \
    mv /tmp/frp_0.61.1_linux_* /tmp/frp; \
    cp /tmp/frp/frpc /usr/local/bin/frpc; \
    chmod +x /usr/local/bin/frpc; \
    rm -rf /tmp/frp /tmp/frp.tar.gz

COPY requirements.txt /
COPY healthcheck.sh /
COPY --chmod=775 start.sh /

ADD li[b] /app/lib

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r /requirements.txt \
 && rm -rf /root/.cache /requirements.txt

WORKDIR /app/lib
ENTRYPOINT ["/start.sh", "python3", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/recognize_backend
HEALTHCHECK --interval=2s --timeout=2s --retries=300 CMD /healthcheck.sh
