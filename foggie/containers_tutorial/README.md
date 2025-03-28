# Example Docker files

This tutorial has Dockerfiles that give some examples of how Docker containers
can be used with [Enzo](https://enzo-project.org/) and [yt](https://yt-project.org/).
Overall, the goal is to make it easier for new users to get up and running with these
codes by creating a "container" (which acts like a virtual machine on your computer)
where Enzo or yt and their dependencies will compile and execute.

The goal of this file is to quickly get you started.  The file
`DOCKER_NOTES.md`, which can be found in this directory, has more
extensive information about using Docker's more complicated features.

**Important terminology:** in this context, a Docker "image" is a read-only template that
has the application code, libraries, and other files that you need to run an application.
A Docker "container" is a running instance of an image, which will be lost when
the container is stopped or deleted. You can run multiple containers using the same
image at the same time, and they are isolated from each other.

## Creating a Docker image for Enzo

1. Download and install [Docker](https://docs.docker.com/get-docker/).
2. At your computer's command line, go to the directory with the
   Dockerfile `Dockerfile.enzo` (which should be the same directory this file is in)
   and type `docker build --build-arg ARCHITECTURE="aarch64" -t enzo-container -f Dockerfile.enzo .` ,
   which uses the Dockerfile in the current directory to build a Linux
   system image and install all of the necessary software (including
   HYPRE, Grackle, and Enzo).  The string `aarch64` tells the Dockerfile that
   your computer is using an ARM chip (i.e., a recent Apple laptop).  Replace that
   string with `x86_64` if you are using a computer with an x86 CPU (any Intel or AMD
   CPU you are likely to use).  Assuming no packages are cached
   locally, this will take 5-20 minutes on a reasonably modern laptop
   and require approximately 1.5 GB of space, which is primarily taken
   up with the various packages that are being installed.  Once this
   is done, you will now have a Docker image called
   `enzo-container`.

## Starting up a Docker container and running Enzo inside it

Once you've created your Docker image, type
`docker run --name enzo-test-container -it enzo-container /bin/bash`
at your system's command line.  This will start up a Docker container
that has a bash command line interface so you can experiment with Enzo,
and will name the container `enzo-test-container`. You do not have to
give the container a name if you don't want to, but Docker will then
generate a random name for you.

You can then run Enzo in the Docker container by doing the following:

```
> cd /sim_data

> enzo -d /root/enzo-dev/run/Hydro/Hydro-1D/SodShockTube/SodShockTube.enzo
```

This will run a simple 1D Sod shock tube, which should take a few
seconds to run and will generate two directories and some files in
the `/sim_data` directory within your Docker container.

Note that when you exit the Docker container (i.e., exit the command
line), if you type the `docker run` command listed above it will start
a NEW container that will not include any changes you've made.  If you
want to reattach the container that you just left, you would type
`docker ps -a` to see the list of existing containers, and then
`docker start <container id> ; docker attach <container id>` to figure
out the hash ID for the container you just exited, restart it, and
then reattach it to your terminal.  Note that `<container id>` is a
long hexadecimal number; you can instead use the container name
(`enzo-test-container` in the example above).

## Creating and using a Docker image to run Enzo at your system command line

Assuming you've installed Docker, at your computer's command line go to
the directory with the Dockerfile `Dockerfile.enzo_cli` (which should be
the same directory this file is in) and type
`docker build --build-arg ARCHITECTURE="aarch64" -t enzo-cli-container -f Dockerfile.enzo_cli .`

(Change `aarch64` to `x86_64` if appropriate for your CPU, as described above.)

After this is done, you now have a Docker image that will let you run Enzo at
the command line.  This will run the containerized version of Enzo (in other words,
using the Enzo binary and all supporting files within the Docker container you've
created, including the Grackle library) at your own computer's command line.
You do this by going into the `enzo_parameter_files` directory in this repository
and typing the following:

```
docker run --rm -v "${PWD}":/share enzo-cli-container -d KelvinHelmholtz.enzo
```

This will use the Enzo binary inside of the container with the Enzo parameter
file in the directory on your own computer.  The `--rm` argument to `docker run`
will remove the Docker container and its internal volumes (but not the original
image!) when enzo is done running.  The `-v "${PWD}":/share` argument maps the
current directory on your computer to a directory `/share` within the Docker
container, which is the internal working directory (i.e., the place where the
containerized Enzo looks for inputs and writes outputs).

**This is one of the most powerful ways to use Docker** because it lets you create
Docker images using whatever compilers, software versions, etc. that you want
that are totally isolated from the rest of your computer, and then run it at your
own command line on data residing on your own computer.  This insulates the code
you've containerized from updates to your own computer's operating system,
compilers, libraries, etc.

As a note, you can set up an alias for the command above.  If you use bash, add
the following line to your `.bashrc` file:

```
alias enzo-docker='docker run --rm -v "${PWD}":/share enzo-cli-container'
```

The `enzo-docker` alias will now include all of the command line arguments, so
you just need to type:

```
enzo-docker -d KelvinHelmholtz.enzo
```

to run the containerized version of Enzo.


## Creating a Docker image with yt and Jupyter notebooks

The file `Dockerfile.yt-jupyter` is a Docker file that uses a different [base
image](https://docs.docker.com/build/building/base-images/) than the Enzo docker
file - rather than using a bare-bones Ubuntu base image, it
uses an image from the [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/),
which are a set of Docker images that contain Jupyter applications and various
useful tools.  In this case we will use the `jupyter/scipy-notebook` base image
and install [yt](https://yt-project.org/), [yt\_astro\_analysis](https://yt-astro-analysis.readthedocs.io/en/latest/), and [Trident](https://trident-project.org/)
on top of it.

Our container is built using essentially the same command as before, which uses the
file `Dockerfile.yt-jupyter` and creates a container called `yt-container` using the
current directory as Docker's working directory:


```
docker build -t yt-container -f Dockerfile.yt-jupyter .
```

Once this is done, we launch the container using the following command:

```
docker run --name yt-test-container -it -v "$PWD":/home/jovyan --rm -p 8888:8888 yt-container
```

This has a bunch of additional arguments beyond what we used for the Enzo docker file.
You can learn about these using the command `docker help run`, but the breakdown of
these arguments are as follows:

* `docker run` - the basic command to run Docker
* `--name yt-test-container` - names the container `yt-test-container`
* `-it` - combines `-i` (interactive mode) and `-t` (terminal mode)
* `-v "$PWD":/home/jovyan` - This bind mounts a volume.  Specifically, it takes the current directory (`$PWD`) and mounts it to the directory `/home/jovyan` inside of the Docker container.  Enclosing `$PWD` in double quotes ensure that it will still work even if your path name has spaces in it.
* `--rm` - This automatically removes the container and its associated anonymous internal volumes when it exits.  This will ensure that the container is not running in the background, but it ALSO means that any changes made within the container will not work.
* `-p 8888:8888` - This publishes the container's 8888 port to the same port on the host.  This is needed by Jupyter-lab so that it can send information to your web browser.
* `yt-container` - this uses the container that you just created.

When you run this it will create a great deal of output, including a link with
a `127.0.0.1` IP that you can copy and paste into your browser.  This will open
up a Jupyter-lab instance in your browser.  Note that the working directory is
the directory you ran Docker in, not a directory inside of the container!  This
means that you can use the software inside of the container to work on data
outside of it.


