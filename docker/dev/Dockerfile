
# General guidelines and recommendations for writing Dockerfile:
# https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/

FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

# Avoid user interaction during install of tzdata
# https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV TZ=Asia
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

################################################################################

# Disable problematic NVIDIA repository to avoid mirror sync issues
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list || true

# Install common essentials required packages/tools/dependencies for subsequent installations.
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    wget \
    locate \
    libglib2.0-0 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libcairo-gobject2 \
    libatk1.0-0 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libfontconfig1 \
    libfreetype6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    libx11-6 \
    libxss1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | bash


# Define the working directory.
WORKDIR /root/dipcom_env/

# Copy the helper scripts to the working directory.
COPY ./docker/dev/scripts/. /root/dipcom_env/docker/dev/scripts/
RUN chmod -R +x /root/dipcom_env/docker/dev/scripts/*

# Setup the Bash shell environment with '/root/.bashrc'.
RUN echo "export MUJOCO_GL='glx'" >> /root/.bashrc
RUN echo "source /root/dipcom_env/docker/dev/scripts/initialize-bash-shell.sh" >> /root/.bashrc


# Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# Traceability.
ARG DOCKERFILE_COMMIT_SHORT_SHA
ENV DOCKER_IMAGE_VERSION_osx_UR=${DOCKERFILE_COMMIT_SHORT_SHA}

# Overwrite the entry point of the parent image.
ENTRYPOINT []

# Enter the container with a Bash shell.
CMD ["/bin/bash"]
