ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.9.0

# Build llama.cpp with both CPU and CUDA backends
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

RUN apt-get update && \
    apt-get install -y gcc-14 g++-14 build-essential git cmake libssl-dev

ENV CC=gcc-14 CXX=g++-14

WORKDIR /build

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git .

# Build with CPU backend variants and CUDA
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_CUDA=ON && \
    cmake --build build -j $(nproc)

RUN mkdir -p /build/lib && \
    find build -name "*.so*" -exec cp -P {} /build/lib \;

# Runtime image with CUDA runtime libraries
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        python3 \
        python3-pip \
        python3-venv && \
    pip3 install --break-system-packages "huggingface-hub[hf_xet]" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=build /build/lib/ /app/
COPY --from=build /build/build/bin/llama-server /app/llama-server

ENV PATH="/app:${PATH}"
ENV LLAMA_ARG_HOST=0.0.0.0

WORKDIR /app

COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

HEALTHCHECK CMD ["curl", "-f", "http://localhost:8080/health"]

EXPOSE 8080

CMD ["/app/start.sh"]
