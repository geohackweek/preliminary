---

title: "Intro and Preparation for Geohackweek"
teaching: 15
exercises: 0
questions:
- "Will my laptop work for this hackathon?"
- "What version of Python should I install?"
- "What tools do I need to participate?"
objectives:
- Getting students ready for running code on their machines during the geohackweek
keypoints:
- we are providing a series of different tools to help you deploy Python efficiently during the hackathon
- Both Python 2.7 and 3.5 will be used for this hackathon
- Docker will be used to work with Google Earth Engine and the other tutorials
- Conda package manager will be used to install python and other libraries

---

This tutorial is a preliminary tutorial that we would like you to complete before arriving at Geohackweek. The purpose is to learn about the minimum system requirement for the Geohackweek, install the necessary softwares used during the week, and hopefully answer any other technical question about your computer setup. Our goal is to have everyone up-and-running prior to the event so we can focus our time more productively when you arrive in person.

### Overview

Python software is distributed as a series of _libraries_ that are called within your code to perform certain tasks. There are many different collections, or _distributions_ of Python software. Generally you install a specific distribution of Python and then add additional libraries as you need them. There are also several different _versions_ of Python. The two main versions right now are 2.7 and 3.5. Some libraries only work with specific versions of Python.

So even though Python is one of the most adaptable, easy-to-use software systems, you can see there are still complexities to work out and potential challenges when delivering content to a large group. Therefore we have a number of different ways that we are trying to simplify this process to maximize your learning during Geohackweek.

Our recommended method is to use [docker](https://www.docker.com/), a system for _containerizing_ computer platforms. We are distributing our tutorial content as a series of these containers. Once installed, you can run Python within these virtual environments on any platform. Even if you are already an experienced Python user, we recommend you using docker. It is a very powerful method for sharing software implementations, and is something worth learning. The challenge with using docker is that it is relatively new and is optimized for specific, modern versions of software and hardware. Those of you with older computers or software versions may encounter challenges in using docker effectively. 

So, we have some backup approaches: we will provide you with access to [SageMathCloud](https://cloud.sagemath.com), a cloud environment that you access through a browser and interface with using Jupyter Notebooks. We'll have more details on that very shortly. We also provide instructions for using [Anaconda](https://www.continuum.io), which is our recommended Python distribution. We can assist in setting up "conda" environments that will simplify the gathering of Python libraries and version specific to the tutorial you are working on.

### Docker: Minimum System Requirements

These are the system requirements necessary to run docker. If your computer does not meet these requirements, please contact us and we can discuss connecting you with other options.

1. Mac OS
    - Mac must be a 2010 or newer model, with Intel’s hardware support for 
    memory management unit (MMU) virtualization; i.e., Extended Page Tables (EPT)
    - macOS 10.10.3 Yosemite or newer
    - At least 4GB of RAM
    - VirtualBox prior to version 4.3.30 must NOT be installed

2. Windows
    - 64bit Windows 10 Pro, Enterprise and Education (1511 November update, Build 10586 or later)
    - Microsoft Hyper-V enabled
    - At least 4GB of RAM

3. Linux
    - Any of the Distro below:
        - Arch Linux
        - CentOS
        - CRUX Linux
        - Debian
        - Fedora
        - Gentoo
        - Oracle Linux
        - Red Hat Enterprise Linux
        - openSUSE and SUSE Linux Enterprise
        - Ubuntu
    - At least 4GB of RAM
       
### Getting set up with Docker

During the Geohackweek, each tutorials will have their own Docker image that 
contains all the tools and libraries you need to go through the tutorial. 
This is especially important for Google Earth Engine. *If you are unable to meet the 
minimum system requirements above, please let us know. For a few of our tutorials, 
is likely that we will group you with others since some content will only be deployed on docker.*

Click [here](https://geohackweek.github.io/preliminary/01-install-docker/) to access the next Episode in this tutorial on installing Docker. Once you have gone through, and Docker has been successfully installed in your machine, or you are having problems, please let us know in Slack.

Continue on to "Getting setup with Conda" once you have gone through setting up with Docker, even if you are unable to install Docker.

### Getting set up with Conda

Conda is a great package management tool, similar to pip. We will be using various
Python libraries with multiple dependencies, so it is critical that you have some sort of 
package management system in place. Conda can be installed in almost any computer.

Here are the system requirements:

- Windows Vista or newer, OS X 10.7+, or Linux (Ubuntu, RedHat and others; CentOS 5+)
- 32-bit or 64-bit
- Minimum 3 GB disk space to download and install.

Click [here](https://geohackweek.github.io/preliminary/02-conda-tutorial/) to start our Conda tutorial. Let us know if on Slack you are having problems with installing Conda.

### Getting setup with Git

Be sure to arrive at Geohackweek with your own [GitHub](https://github.com/) account.

### Getting setup with Sage Math Cloud

[SageMathCloud](https://cloud.sagemath.com) (SMC) is a cloud computing platform that allows you to create "projects", each of which is a virtual machine running Linux that you can access from any web browser. A lot of scientific software is pre-installed on any project, including the Anaconda stack, IPython, and Jupyter notebooks, and also has other nice tools such as a WYSIWIG Latex editor, for example.

We have created a project that has much of the software needed for the tutorials already installed on it, and we can push a clone of this project to any participant who wants to use it.  If you have a hard time getting some of the software to work on your laptop, this will provide another option for the week.

SMC also makes it easy to collaborate with other participants since you can easily add collaborators to a project and work on the same set of files.

In order to use SMC, the first step is to create a free account on the [SageMathCloud](https://cloud.sagemath.com) site.

If you would like a clone of the project please send a direct message to @rjleveque via [Slack](https://geohackweek2016.slack.com) with the email address used to create your SMC account.
 
Note that we have purchased a "course plan" from SMC so that each of these projects will be upgraded to "member hosting" (running on a faster server) and will also have internet access (so you can transfer data in and out).  These upgrades will be in effect for 4 months.  Your project will be available "forever" via your free account.  With a [paid subscription](https://cloud.sagemath.com/policies/pricing.html) you can upgrade your account to also maintain member hosting and internet access in the future.

### Creating an account for Google Earth Engine
In order to use Google Earth Engine, you need to sign up for the platform 
and be approved as a trusted tester. Getting approved can take 2-3 days, 
so please sign up no later than November 8th to make sure you are approved in time. 
Click [here](https://geohackweek.github.io/GEE-Python-API/00%20-%20GEE%20Access/) and follow the direction to sign up.



