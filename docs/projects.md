Projects
========

Each aframe project represents an individually containerized sub-task of the aframe analysis. This allows project environments
to be lightweight, flexible and portable. Projects are meant to produce _artifacts_ of some specific experiment. By artifact, we typically mean some file (e.g training data, optimized models, analysis plots, etc.) saved to disk somewhere. Projects should be kept modular and specific to the artifact they are designed to generate.

## Current Projects
```{eval-rst}
.. toctree::
   :glob:
   :titlesonly:
   :maxdepth: 1
   
   projects/*
```

## Building Containers
Most projects are fully python based, and their environments are managed using [uv](https://docs.astral.sh/uv/).
The `data` and `online` project also requires the use of [`Mamba`](https://mamba.readthedocs.io/en/latest/).

In the root directory of (most) projects is an `apptainer.def` file that containerizes 
the project application using [Apptainer](https://apptainer.org/docs/user/latest/).

Before building project container images, it is recommended that you set the `AFRAME_CONTAINER_ROOT` environment variable.
A good location for this would be in your `~/.bash_profile` (or whichever shell-dependent scripts are automatically run when you login)

```bash
echo export AFRAME_CONTAINER_ROOT=~/aframe/images/ >> ~/.bash_profile
mkdir -p $AFRAME_CONTAINER_ROOT
```

This is the location where `aframe` images will be stored. `luigi`/`law` tasks will look for images in this location.

The root `aframe` environment ships with a command line utility for building all of the project containers in parallel

```bash
uv run build-containers
```
By default, this will store the images in the location specified by the `AFRAME_CONTAINER_ROOT` environment variable.
If you only want to build certain project containers, you can specify their names as arguments.

```bash
uv run build-containers data export 
```

## Executing a Container
Most projects come with a command line interface (CLI) built with the extremely flexible [jsonargparse](https://jsonargparse.readthedocs.io/en/stable/). As an example of how you might run a command inside the container, let's use the CLI in the data container to query for science-mode segments from [gwosc](https://gwosc.org/)

```bash
mkdir ~/aframe/data/
apptainer run $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data query --flags='["H1_DATA", "L1_DATA"]' --start 1240579783 --end 1241443783 --output_file ~/aframe/data/segments.txt
```

See each projects README for details on available commands.

## Rebuilding Containers
If a project requires dependencies to be changed, modified or added, the corresponding container will have to be rebuilt to reflect that. 
However, code-only changes can be mapped from your local filesystem into the container at runtime using the `--bind` flag:

```bash
apptainer run --bind .:/opt/aframe $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data query --flags='["H1_DATA", "L1_DATA"]' --start 1240579783 --end 1241443783 --output_file ~/aframe/data/segments.txt
```

The above command will bind your current working directory to `/opt/aframe` inside the container.
Make sure you are binding the root of the `aframe` repository to `/opt/aframe`, the location where the repository code is located inside all `aframe` containers. 


## Building Containers by Hand
It may be useful at times to know how to build containers by hand. 
As an example, let's build the `data` container image. Once inside the `data` root directory, building the image is as simple as 

```bash
apptainer build $AFRAME_CONTAINER_ROOT/data.sif apptainer.def
```

```{eval-rst}

.. note: 
    It is highly recommended that you name containers after the corresponding project. Although not strictly necessary, this is the default expecatation of the `luigi`/`law` tasks
```

Each projects `README` has instructions for building its container/environment.


## Tips and Tricks
For development, it can often be useful to open a shell inside a container to poke around and debug: 

```bash
apptainer shell $AFRAME_CONTAINER_ROOT/image.sif
```

To allow GPU access, simply add the `--nv` flag to your `apptainer` command. Also, environment variables with the `APPTAINERENV_` prefix
will automatically be mapped into the container:

```bash
APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1 apptainer run --nv $AFRAME_CONTAINER_ROOT/train.sif ...
```
