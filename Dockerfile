# Use an official base image with Conda, such as ContinuumIO miniconda image
FROM continuumio/miniconda3

# Set a working directory inside the container
WORKDIR /app

# Copy the environment.yml file into the container
COPY environment.yaml /app/


# add huggingface token as env
ENV HF_TOKEN=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

# Install the conda environment specified in environment.yaml
RUN conda env create -f environment.yaml

# Make sure conda is initialized for shell usage
RUN echo "conda activate gigapath" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Activate the environment and install dependencies using pip
# Copy the current directory (project code) into the container
COPY . /app

# Install the package in editable mode
RUN conda activate gigapath && pip install -e . && pip install jupyterlab && pip install open_clip_torch

# install python3-openslide
RUN apt-get update --fix-missing
RUN apt-get install -y python3-openslide
RUN apt install graphviz -y
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0


# Set environment variable to ensure the conda environment stays active
ENV PATH /opt/conda/envs/gigapath/bin:$PATH

# install packages in the pkgs directory
COPY ./pkgs/ ./pkgs/
RUN for d in ./pkgs/*/ ; do echo "Installing package" "$d" ; pip install "$d"; done

# Ensure the environment is activated when the container starts
CMD ["bash", "-c", "source activate gigapath && exec bash"]
