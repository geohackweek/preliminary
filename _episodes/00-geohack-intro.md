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
- Linux or Mac will be the best choice for geohack, though windows would work
- Both Python 2.7 and 3.5 will be used for this hackathon
- Docker will be used to work with Google Earth Engine and the other tutorials
- Conda package manager will be used to install python and other libraries

---

This tutorial is for you to get set up for the Geohackweek. By following these steps, you’ll 
be learn about the minimum system requirement for the Geohackweek, install the necessary 
softwares used during the week, and hopefully answer any other technical question about your computer setup.


### Minimum System Requirements
We recommend that you work with either Linux or Mac, though it is okay to use the newest Windows 10.

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
    Arch Linux
    CentOS
    CRUX Linux
    Debian
    Fedora
    Gentoo
    Oracle Linux
    Red Hat Enterprise Linux
    openSUSE and SUSE Linux Enterprise
    Ubuntu
    - At least 4GB of RAM
    
    
### Getting setup with Docker
During the Geohackweek, each tutorials will have their own Docker image that 
contains all the tools and libraries you need to go through the tutorial. 
This is especially important for Google Earth Engine. *If you are unable to meet the 
minimum system requirements above, please let us know. It is likely that you will have
to work with others.*

In order to get setup with Docker, click on the Episodes in the navigation bar and go to 
Installing Docker. Once you have gone through, and Docker has been successfully installed
in your machine, or you are having problems, please let us know in Slack.

Continue on to Getting setup with Conda once you have gone through setting up with Docker,
even if you are unable to install Docker.

### Getting setup with Conda
Conda is a great package management tool, similar to pip. We will be using various
Python libraries with multiple dependencies, so it is critical that you have some sort of 
package management system in place. Conda can be installed in almost any computer.

Here are the system requirements:
- Windows Vista* or newer, OS X 10.7+, or Linux (Ubuntu, RedHat and others; CentOS 5+)
- 32-bit or 64-bit
- Minimum 3 GB disk space to download and install.

In order to get setup with Conda, click on the Episodes in the navigation bar and go to 
Getting started with Conda. Once you have gone through, and Conda has been successfully installed
in your machine, or you are having problems, please let us know in Slack.

Continue on to Getting setup with Git once you have gone through setting up with Conda.

### Getting setup with Git


### Getting setup with Sage Math Cloud


### Creating an account for Google Earth Engine
In order to use Google Earth Engine, you need to sign up for the platform 
and be approved as a trusted tester. Getting approved can take 2-3 days, 
so please sign up no later than November 8th to make sure you are approved in time. 
Click [here](https://geohackweek.github.io/GEE-Python-API/00%20-%20GEE%20Access/) and follow the direction to sign up.

### Installing Python
During the Geohackweek, each tutorials will have their own Docker image that 
contains the correct version of Python to go through the tutorial. *If you do not have Docker
or have never encounter this term, please proceed to the Docker preliminary tutorial to get 
yourself familiar and set up.*

We will be using both Python 2.7 and Python 3.5 during the week. Though the two versions
of python are quite similar, they have some syntax differences which you can take a look [here.](#)

If you are new or rusty on Python, please review Python 3.5 since the syntax is
backwards compatible to Python 2.7.

In order to install Python in your local machine, please use the Conda package manager, 
which will save you a lot of headaches down the road when trying to install other
libraries. *Please go through the Conda preliminary tutorial to get you set up.*



