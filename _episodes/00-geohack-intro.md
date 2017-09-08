---

title: "Intro and Preparation for Geohackweek"
teaching: 15
exercises: 0
questions:
- "Will my laptop work for this hackathon?"
- "What version of Python should I install?"
- "What tools do I need to participate?"
objectives:
- Getting patricipants ready for running code in the cloud and on their local machines during the geohackweek
keypoints:
- participants will conduct tutorial exercises in a cloud environment
- everyone is encouraged to arrive with Python installed on their laptop for the project work
- Both Python 2.7 and 3.6 will be used for this hackathon
- Conda package manager will be used to install python and other libraries

---

This preliminary tutorial is one that we would like you to complete before arriving at Geohackweek. The purpose is to learn about the minimum system requirement for the Geohackweek, install the necessary software used during the week, and hopefully answer any other technical question about your computer setup. Our goal is to have everyone up-and-running prior to the event so we can focus our time more productively when you arrive in person.

### Overview

Python software is distributed as a series of *libraries* that are called within your code to perform certain tasks. There are many different collections, or *distributions* of Python software. Generally you install a specific distribution of Python and then add additional libraries as you need them. There are also several different *versions* of Python. The two main versions right now are 2.7 and 3.6. Some libraries only work with specific versions of Python.

So even though Python is one of the most adaptable, easy-to-use software systems, you can see there are still complexities to work out and potential challenges when delivering content to a large group. Therefore we have a number of different ways that we are trying to simplify this process to maximize your learning during Geohackweek.

This year, we will be using [JupyterHub](https://jupyterhub.readthedocs.io/en/latest/) hosted on [Amazon Web Services (AWS) Cloud](https://aws.amazon.com/). JupyterHub is a multi-user Hub that provides multiple instances of the single-user Jupyter notebook server. A Jupyter Notebook is an open-source web application that allows users to create and share documents containing live code, equations, visualizations, and markdown texts. Here's a [overview](https://www.slideshare.net/willingc/jupyterhub-a-thing-explainer-overview?from_action=save) on JupyterHub.

We also provide instructions for using [Anaconda](https://www.continuum.io), which is our recommended Python distribution. We can assist in setting up "conda" environments that will simplify the gathering of Python libraries and version specific to the tutorial you are working on.

### Getting set up with Conda

[**Conda**](http://conda.pydata.org/docs/) is an **open source `package` and `environment` management system for python libraries**. We will be using various
Python libraries with multiple dependencies, so it is critical that you have some sort of 
package management system in place. Conda can be installed in almost any computer.

Here are the system requirements:

- Windows Vista or newer, OS X 10.7+, or Linux (Ubuntu, RedHat and others; CentOS 5+)
- 32-bit or 64-bit
- Minimum 3 GB disk space to download and install.

Click [here](https://geohackweek.github.io/preliminary/01-conda-tutorial/) to start our Conda tutorial. Let us know if on Slack you are having problems with installing Conda.

### Getting setup with Git

Be sure to arrive at Geohackweek with your own [GitHub](https://github.com/) account.

### Getting setup with JupyterHub on AWS

For those who have their Github accounts posted on the [Geohackweek Github Wiki](https://github.com/geohackweek/ghw2017/wiki/Participants), you can try accessing the Jupyterhub server we will be using throughout the program by clicking [here](https://jupyterhub.cloudmaven.org). Make sure you are able to select the Anaconda kernel on the top right corner when you click on the "New" button. You will be logging into Jupyterhub using your Github credentials.

### Creating an account for Google Earth Engine
In order to use Google Earth Engine, you need to sign up for the platform 
and be approved as a trusted tester. 
Click [here](https://geohackweek.github.io/GoogleEarthEngine/00-access-javascript/) and follow the direction to sign up.

### Brushing up on Python

Given all the geospatial content we hope to cover during geohack week, we will not be able to provide instruction in Python fundamentals. Here are some excellent resources we recommend using to brush up on your Python skills in advance of our event:

* [Software carpentry lessons](https://software-carpentry.org/lessons/), especially the [Python tutorial](http://swcarpentry.github.io/python-novice-inflammation/)
* [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) 


