# Some notes on using Docker

The goal of this document is to complement the `README.md` file found
in this directory.  It provides a brief introduction to Docker and to
a variety of Docker commands, and suggests potential workflows that
might be useful as you work with Enzo in a container.

## What is Docker and why is it useful?

[Docker](https://www.docker.com/) is a platform that allows developers
to package and execute applications in an environment called a
"container," which is isolated in critical ways from the system that
it runs on.  In particular, containers can have their own operating
system, software, and supporting data, and are isolated from the host
machine's OS.  This is an important point - it means that containers
provide a consistent and reliable environment in which one can develop
software.  This is useful for software that has dependencies that are
non-trivial to compile and run on particular operating systems (e.g.,
Charm++ or Kokkos on macOS), where you want to experiment with new
compilers or libraries, or where you simply want to isolate what you're
doing from the rest of your system.  It's also very useful as a teaching
tool, as it makes it relatively straightforward for people to get a new
piece of software installed on their system. There are also other use
cases as well!

It is worth noting that the terms "container" and "virtual machine"
are sometimes used interchangeably, but
[they are not the same thing](https://www.backblaze.com/blog/vm-vs-containers/).
The crucial difference is that virtual machines (VMs) typically
virtualize (i.e., emulate) an entire computer including the hardware
and run processes within that virtual machine, whereas containers
require only the software necessary to run a specific program (e.g.,
the necessary components of an OS, libraries, compilers, etc.) and
then run processes directly on the host computer.  This generally
means that containers require fewer system resources than VMs in terms
of disk space and memory, are much quicker to start up, and generally
are more performant. This also means that you can run more containers
than VMs on a given host machine.

**Potentially useful resources:**

* [Intro to Docker, Docker containers, and Virtual Machines (Video; ~9 minutes)](https://www.youtube.com/watch?v=JSLpG_spOBM)
* [Why Docker? (Text; Basically a sales pitch, but informative)](https://www.docker.com/why-docker)
* [Docker Tutorial for Beginners (Video; ~2 hours)](https://www.youtube.com/watch?v=fqMOX6JJhGo)
* [Docker documentation (text)](https://docs.docker.com/)
* [Getting started with docker (text; within Docker docs)](https://docs.docker.com/get-started/overview/)

## Getting help

In addition to the [Docker documentation](https://docs.docker.com/)
(which is extensive and high-quality) and Stack Overflow's
[Docker section](https://stackoverflow.com/questions/tagged/docker)
(which is incredibly helpful), Docker itself has great command-line
help.  You can type:

```
docker --help
```

to get a listing of Docker options and commands.  If you want to know
more about a specific command, including optional flags or arguments,
you can type:

```
docker COMMAND --help
```


## The Dockerfile, Docker images, and Docker containers

A Dockerfile is a text file that contains the instructions required to
build a Docker image, including the desired operating system
components, libraries and other tools, and applications.  It's useful
for users to have this as a text file rather than a binary Docker
image, because (1) it's much smaller (kilobytes as opposed to 100s of
MB to ~1 GB) and it also allows you to easily update parts of the file
to, e.g., use an additional software package, change the base
container, or point to a different branch or fork of the code you're
working on.

A Docker "image" is a read-only template that has the application code,
libraries, and other files that you need to run an application, and is
generated by building a Dockerfile.

A Docker "container" is a running instance of a Docker image, which will
be lost when the container is stopped or deleted. You can run multiple
containers at the same time, and they are completely isolated
from each other.  This can include multiple instances of the same image,
instances of different images (containing different software, operating
systems, compilers, libraries, etc.), or any combination thereof. This is
a **very** useful capability for code development and testing!

### Building a Docker image

You can build a Docker image using a Dockerfile on your own computer
by downloading a Dockerfile,
[installing Docker](https://docs.docker.com/get-docker/), moving to
the directory where the Dockerfile exists at the command line, and
then typing the following:

```
docker build --build-arg ARCHITECTURE="aarch64" -t enzo-image -f Dockerfile.enzo .
```

This will create a Docker image that is tagged `enzo-image`, will
use the file `Dockerfile.enzo` to create said Docker image, and will
use the current directory (`.`) as Docker's working directory.  The
`--build-arg ARCHITECTURE="aarch64"` string is a command-line argument
for the Dockerfile that indicates the system's hardware architecture.
`"aarch64"` indicates that it is an ARM CPU (i.e., modern Apple laptops).
Replace that with `"x86_64"` if your CPU is Intel or AMD x86 chip.

If you do not use the `-f Dockerfile.enzo` argument
Docker will look for a file named `Dockerfile`, which is the default file.
Note that you can specify other directories as the working directory,
Dockerfiles in different directories, and other things as well. Just
type `docker build --help` for more information.

The above command can take several minutes to run, depending on the
speed of your system and your network connection.  Once it is done you
will have a Docker image named `enzo-image`, which is now ready
to run!

**Potentially useful resources:**

* [What is a Dockerfile?  (blog post)](https://www.cloudbees.com/blog/what-is-a-dockerfile/)
* [Dockerfile reference docs](https://docs.docker.com/engine/reference/builder/)
* [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [Docker Hub - a curated catalog of base images](https://hub.docker.com/)


## Running a Docker container

You can use your new Docker image to start up a container and get bash
command line access to it by typing the following at your system's
command line:

```
docker run -it enzo-image /bin/bash
```

The `-it` flag is equivalent to `-i -t`, and tells Docker that this is
an interactive container (`-i`) and allows you to connect to it via a
terminal (`-t`).  Adding `/bin/bash` after the container name starts
up the container with a bash command line.  (Note that you can also
execute a program contained within the container at your system's
command line, so that it does things on your own computer - see the
`README.md` file for an example!)

Once you have done this you can then run Enzo in this container, as
described in `README.md`.  Note that when you exit the Docker
container (i.e., exit the bash shell you have just started), **this
does not actually delete the Docker container you've started** - it
just detaches that container from the command line and stops it!  If
you execute the `docker run` command listed above another time it will
start a NEW container that will not include any changes you've made;
you will then have two containers running using the same system image.
You can also have multiple containers running using multiple system
images - just make sure to give them distinct names!  (`docker run`
has a `--name <my_chosen_name>` option, which is extremely handy for
this purpose - otherwise, Docker generates a random and not
particularly useful name for each container.)

Note that there are good reasons to do this deliberately - for
example, you may want to have containers with different operating
systems, different versions of the code you're working with, etc. for
development purposes.  However, it's useful to give your containers
distinct names with the `--name` option so that you don't have to
guess which container is which!

If you want to reattach a stopped (but not killed) container to your
terminal, you can type `docker ps -a` to see the list of existing
containers at the command line.  Once you have found the container ID
of the container you wish to reattach, you can type `docker start
<container ID> ; docker attach <container ID>` to restart the
container and then reattach it to your terminal.  You can also start a
container and allow it to run in the background without reattaching it
by just using the first command (`docker start <container ID>`).

You can stop a running container with the `docker stop <container ID>`
command, which will attempt to gracefully stop the processes running
within it before stopping the container after 10 seconds.  If that
doesn't work, you can also kill the container with `docker kill
<container ID>`, which is analogous to the Unix/Linux `kill -9
<process ID>` command and stops it instantly.  In both cases the
container will still exist, but will no longer be running.

You can remove a stopped container with the `docker rm` command.
First, you should type `docker ps -a` to get a list of all containers
currently on the system.  Once you've identified the container ID of
the one you wish to remove, type `docker rm <container ID>` and it
will remove that container.

**It is crucial to understand what happens to data in Docker
  containers,** as this is the origin of many issues that people have
  with containers. If you stop (or pause) a Docker container, you will
  not lose any data that is written to disk within that container.
  **If you remove a Docker container, any changes from the original
  image will be lost!**

**Potentially useful resources:**

* [Docker command line reference](https://docs.docker.com/engine/reference/commandline/docker/)


## Managing data and data movement

By default, processes running in Docker containers can only see data
inside of that container. While this is generally a useful property,
there are many reasons that one might need to share files or
directories between the container and its host computer.  There are
several ways to do this, as described below.

### Copying data

If you want to copy data into or out of a container, you use the
[`docker cp`](https://docs.docker.com/engine/reference/commandline/cp/)
command.  Here's the syntax for copying from the directory SRC\_PATH in
the container to DEST\_PATH on the host computer:

`docker cp <container id>:SRC_PATH DEST_PATH`

You can also do this the other way around, and copy from the host to
the container.  Container paths are relative to the container's root
directory, and host paths can be relative or absolute.  Docker's cp
command copies recursively by default.

Note that `docker cp` does not currently support wildcards, so if you
want to move around multiple files the easiest thing to do is put them
in a directory and copy that.  You can also use the tar command to
copy multiple files into a running container (see
[this example](https://stackoverflow.com/questions/22907231/how-to-copy-files-from-host-to-docker-container)
on Stack Overflow).

### Mounting directories

If you are going to be using your Docker container for significant
amounts of experimentation (as opposed to setting it up to execute at
your own system's command line), you will probably want an easier way to
move data back and forth.  One way to do this is to use a
[bind mount](https://docs.docker.com/storage/bind-mounts/) to create a
directory that exists on both the host computer and in the Docker
image.  You have to actually make the local directory before you make
a bind mount, and then you point toward it when you run the Docker
image.  An example of how to do so is below, which makes a new
directory `enzo-data` in your current working directory (but it does
not have to be there, it can be wherever you want) and then starts the
container we've already created and creates the directory `/enzo-data`
inside of it:

```
mkdir enzo-data

docker run -it --name enzo-mount --mount type=bind,source="$(pwd)"/enzo-data,target=/enzo-data enzo-image /bin/bash
```

Note that the `"$(pwd)"` portion of the command line above will work
for bash/zsh, but probably not behave correctly for tcsh/csh (or other
similar shells).  If you get an error, use the complete path.  Any
data copied into the host computer's enzo-data diretory or into the
container's `/enzo-data` directory will appear in the other one, so
you can move data back and forth.  If you modify Enzo's parameter
files to point to the correct place, you can also write data directly
to the `/enzo-data` directory within the Docker image, and have it
appear on your local machine.

Note that **mounting a non-empty directory on the container will have
unintuitive behavior**.  The existing contents of the directory within
the container are hidden by the bind mount, which is at best confusing
and at worst will break the container or render it non-functional for
your purposes.  You're much better off doing things the other way
around - i.e., mounting non-empty directories on the host into empty
directories on the container.

**It is possible to do code development on your local host but compile
  and execute your code in a container**.  For example, if your copy
  of Enzo is located in the directory `/User/yourname/enzo-dev` on the
  host computer, the following command line will mount that directory
  (and all of its subdirectdories) into the directory
  `/host-enzo-dir` in the container:

```
docker run -it --name enzo-mount --mount type=bind,source=/User/yourname/enzo-dev,target=/host-enzo-src enzo-image /bin/bash
```

Once you do this, any files that are edited on the host machine in
`/User/yourname/enzo-dev` or its subdirectories will be visible in the
container `/host-enzo-src`, and you can compile and execute the
source code from within the container.  Note that in order to do so,
you will have to edit Enzo's makefile to point to the correct paths
within the container!  (Note that you can do this to, for example,
compile Enzo using Linux while working on an Apple laptop!)

**It is also possible to mount multiple directories between the host
  and container**.  This is useful if you want to have your code in
  one directory and output simulation data into another directory, for
  example.  This is done by having multiple calls to `--mount` in your
  `docker run` command.  Building on the example directly above, let's
  assume that we want to read or write data in a directory on the host
  called `/User/yourname/enzo-data` and mount it into the container
  in the directory `/enzo-data`.  We also wish to use the same
  source code directory as we did before.  To do so, we simply call
  the `--mount` option twice:

```
docker run -it --name enzo-mount --mount type=bind,source=/User/yourname/enzo-dev,target=/host-enzo-src --mount type=bind,source=/User/yourname/enzo-data,target=/host-enzo-data enzo-image /bin/bash
```

This can in principle be done with as many directories as you want.
Furthermore, **note that a given directory on the host machine can
only be used by a single container at a time!**

### Docker Volumes

Docker [Volumes](https://docs.docker.com/storage/volumes/) are a more
fully-featured solution than bind mounts, and work reasonably well on
both Linux and Windows machines (though not nearly as well on macOS/OS
X).  Volumes are overkill for the likely use cases for Enzo
developers, which can likely be taken care of with bind mounts.  To
that end, the interested reader should be aware of their existence and
consult the link above if they want to experiment.

**Potentially useful resources:**

* [Documentation for `docker cp`](https://docs.docker.com/engine/reference/commandline/cp/)
* [Documentation for bind mounts](https://docs.docker.com/storage/bind-mounts/)
* [Documentation for volumes](https://docs.docker.com/storage/volumes/)


## Cleaning up after yourself

While Docker is useful, it can leave a lot of detritus on the system.
For example, if you run the same Dockerfile more than once you may
notice that many of the installed packages are cached locally, which
may be undesirable if you are testing new or modified Dockerfiles.
You may also find that you have multiple old images on the system, or
other Docker resources such as networks.  To delete dangling resources
(i.e., those that are not associated with a running container) type
the following at the command line:

```
docker system prune
```

You can remove everything (including effectively all resources not
currently being used, including volumes), with:

```
docker system prune -a --volumes
```

**Be very careful with pruning your system!**  If you just want to
  remove a single system image, you can get a list of all of your
  Docker images with:

```
docker images -a
```

and then remove the image you wish to get rid of with:

```
docker rmi <IMAGE ID>
```

If Docker argues with you, you can force this with the `-f` argument
(but should be cautious about doing so).

**If you only wish to remove a stopped container** (so that you can
  start another container using host directories bound to it, for
  example), you can do this by first identifying the container ID with
  `docker ps -a`, and then removing it with `docker rm <container
  ID>`.

**Potentially useful resources:**

* [system prune documentation](https://docs.docker.com/engine/reference/commandline/system_prune/)
* [docker system documentation](https://docs.docker.com/engine/reference/commandline/system/)
* [Tutorial on removing Docker images and volumes](https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes)



## Some other information about using Docker containers

### Visual Studio Code

If you use [Visual Studio Code](https://code.visualstudio.com/), it
has an extension that will allow you to
[develop inside a Docker container](https://code.visualstudio.com/docs/remote/containers)
using VSCode's full feature set. See [this YouTube video](https://www.youtube.com/watch?v=bUhjY2L1iFc)
for an overview explaining how to get started with the VS Code Docker extension.

### Singularity and Podman

While Docker is incredibly useful, most supercomputing centers do not
allow users to run Docker containers through Docker itself because it
requires root access - i.e., more priveleged access to the machine than
supercomputer system administrators are willing to share (for security
and system stability reasons, among others). We are going to consider two of these tools, [Podman](https://podman.io/) and [Singularity](https://sylabs.io/).  Podman runs on NASA's Aitken cluster, and Singularity runs on MSU's ICER clusters.

**Running on NASA Pleiades**

[Podman](https://podman.io/) is an open-source set of container tools that allow "rootless containers" - this means that, unlike Docker, you don't need root access to a system in order to provide encapsulation in the way that Docker does.  It also can run on multiple cores, multiple nodes, and on GPUs, and can (most importantly for us!) use Docker image files.  See their [Documentation](https://podman.io/docs) and [Getting Started](https://podman.io/get-started) pages for more information.

Podman is the preferred container tool of choice on NASA's computers for a variety of reasons ([detailed here](https://www.nas.nasa.gov/hecc/support/kb/why-is-podman-recommended-for-use-at-nas_699.html)).  There are some limitations to using it on the NASA HEC systems: you can only run on a single node at a time (though you can use MPI within that node), and it will only run on the [Aitken](https://www.nas.nasa.gov/hecc/support/kb/aitken-configuration-details_580.html) AMD EPYC [Milan](https://www.nas.nasa.gov/hecc/support/kb/amd-milan-processors_688.html) and [Rome](https://www.nas.nasa.gov/hecc/support/kb/amd-rome-processors_658.html) nodes, which have have 2 64-core CPUs per node (128 cores total).  Some useful references from the HECC knowledge base are as follows:

* [Getting Started with Podman at NAS](https://www.nas.nasa.gov/hecc/support/kb/getting-started-with-podman-at-nas_698.html)
* [Preparing a Podman Image for use on NAS systems](https://www.nas.nasa.gov/hecc/support/kb/preparing-a-podman-image-for-use-on-nas-systems_697.html)
* [Running Processes in a Podman Container](https://www.nas.nasa.gov/hecc/support/kb/running-processes-in-a-podman-container_696.html)
* [List of all NAS Knowledge Base Podman pages](https://www.nas.nasa.gov/hecc/support/kb/podman-190/)

**Running at Michigan State University**

[Singularity](https://sylabs.io/) is an alternative to Podman.  it is a container environment that
works on supercomputers, allows for performant multi-core, multi-node,
and GPU execution, provides similar encapsulation to Docker, and (most
importantly for us!) can use Docker image files.  See the
[Singularity documentation](https://sylabs.io/docs/) for documentation
about Sinularity, and consult your local supercomputing center's
documentation for instructions on how to load and use it there.

MSU's [Institute for Cyber-Enabled Research](https://icer.msu.edu) has
documentation that gives examples of how to transition from Docker
to Singularity on the MSU HPCC.  While the details of these examples
are specific to ICER, the principles should generalize to other
supercomputers that run Singularity:

* [Containers overview](https://docs.icer.msu.edu/Containers_Overview/)
* [Docker overview](https://docs.icer.msu.edu/Docker/) - note that this is somewhat duplicative of this document.
* [Introduction to Singularity](https://docs.icer.msu.edu/Singularity_Introduction/)
* [Singularity Overlays](https://docs.icer.msu.edu/Singularity_Overlays/) - Overlays are a way to enapsulate many small files (like an Anaconda python installation) inside of a single large file for performance reasons.  Note that this documentation is specific to ICER's clusters, but the Singularity website has more general [documentation on overlays](https://docs.sylabs.io/guides/3.2/user-guide/persistent_overlays.html?highlight=overlay) that can be applied anywhere.
* [Advanced Singularity topics](https://docs.icer.msu.edu/Singularity_Advanced_Topics/) - this includes building your own containers, migrating from Docker to Singularity, and using Singularity with MPI and GPUs.
