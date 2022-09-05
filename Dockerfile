FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "create_basin_mask", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flake8"

# The code to run when container is started:
COPY . /app
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "create_basin_mask", \
            "python", "convert_shp_to_raster.py", \
            "./data/input/Indus.dir/Indus.shp",  "--res=0.5"]