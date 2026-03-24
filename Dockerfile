FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install uv directly from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies via uv.
# We strip out torch, torchvision, torchaudio, tbb, triton and nvidia-* because the NGC PyTorch image already provides highly-optimized versions.
# Remove the EXTERNALLY-MANAGED marker so uv/pip can install system-wide in the container.
# Strip strict version pins (==x.y.z) since requirements.txt was created for Python 3.8
# and many pins are incompatible with Python 3.12. We keep >=/>/>= constraints.
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    grep -v "^nvidia-" requirements.txt | grep -v "^torch" | grep -v "^tbb" | grep -v "^triton" | grep -v "^tensorflow" | grep -v "^keras" | \
    sed 's/==.*//' > req_filtered.txt && \
    uv pip install --system -r req_filtered.txt && \
    uv pip install --system "transformers>=4.40,<5"

# Copy the rest of the project
COPY . .

RUN chmod +x /app/scripts/container-entrypoint.sh

# Arch-aware OpenMP preload (aarch64 only); see scripts/container-entrypoint.sh
ENTRYPOINT ["/app/scripts/container-entrypoint.sh"]

# Set the default command to bash
CMD ["bash"]
