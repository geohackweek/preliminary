---

title: "Getting Started with Conda"
teaching: 15
exercises: 0
questions:
- "What is Conda?"
- "How do I install Conda?" 
objectives:
- Get conda installed!
keypoints:
- conda can be installed in two ways (Anaconda and Miniconda)
- conda package manager works across systems

---

### What is Conda?
Similar to [pip](https://pypi.python.org/pypi/pip), [**conda**](http://conda.pydata.org/docs/) is an **open source package and environment management system** for installing multiple versions of software packages, their dependencies and switching easily between them. It works on Linux, OS X and Windows.

### What is Anaconda?
[Anaconda](https://www.continuum.io/why-anaconda) is a data science platform that comes with a lot of packages. At the core, Anaconda uses conda package management system.

- List of packages included can be found [*here*](https://docs.continuum.io/anaconda/pkg-docs)

- If you don't have time or disk space for the entire distribution, try [Miniconda](http://conda.pydata.org/miniconda.html), a bootstrap version of Anaconda, which contains only Python, essential packages, and conda. The rest of the packages has to be installed individually.

- [This Continuum blog post is a terrific, recent and comprehensive introduction to conda](http://www.continuum.io/blog/conda-data-science) targeted to data scientists. It also has links to a presentation (Youtube and slides) on the same material. An extra nifty aspect of this material is that it "explores how to use conda in a multi-language data science project" with an example combining Python and R libraries.

### Installing Anaconda

1. To install Anaconda, please click on the link below for your operating system, and follow the instructions on the site:
  - [Windows](https://www.continuum.io/downloads#windows)
  - [OSX](https://www.continuum.io/downloads#osx)
  - [Linux](https://www.continuum.io/downloads#linux)
2. Once Anaconda installation step is finished run `python` in the command line to test if Anaconda is installed correctly.
If Anaconda is installed correctly, you should have this prompt, which emphasizes **Anaconda**:

```bash
$ python
Python 2.7.11 |Anaconda custom (x86_64)| (default, Dec  6 2015, 18:57:58)
[GCC 4.2.1 (Apple Inc. build 5577)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
Anaconda is brought to you by Continuum Analytics.
Please check out: http://continuum.io/thanks and https://anaconda.org
>>>
```

### Installing Miniconda

##### Windows
Navigate to http://conda.pydata.org/miniconda.html and download the proper installer for you Windows platform (32 or 64 bits).
We recommend to download the Python 2 version of Miniconda as not all packages are Python 3-compliant yet.  You can still create new Python 3 environments using the Python 2 verson of Miniconda, so you are not limiting yourself.

When installing you will be asked if you wish to make the Anaconda Python your default Python for Windows.
If you do not have any other installation that is a good option.  If you want to keep multiple versions of python on your machine (e.g. ESRI-supplied python, or both 32 and 64 bit versions of Anaconda), then don't select the option to modify your path or modify your windows registry settings.

##### Linux and OSX
You may follow manual steps from http://conda.pydata.org/miniconda.html similar to the instructions on Windows (see above). Alternatively, you can execute these commands on a terminal shell (in this case, the bash shell):

```bash
url=http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
curl $url -o miniconda.sh
bash miniconda.sh -b
export PATH=$HOME/miniconda2/bin:$PATH
conda update --yes --all
```

### Installing Python
During the Geohackweek, each tutorials will have their own Docker image that  
contains the correct version of Python to go through the tutorial. 
*If you do not have Docker  or have never encounter this term, 
please proceed to the Docker preliminary tutorial to get  yourself familiar 
and set up.*

  We will be using both Python 2.7 and Python 3.5 during the week. 
Though the two versions  of python are quite similar, they have some syntax 
differences which you can take a look [here](http://sebastianraschka.com/Articles/2014_python_2_3_key_diff.html). 

Using Conda, you can install both Python 2.7 and 3.5 using separate environments 
(Details will be explained during the Introductory tutorial).

#### Installing Python 2.7
`$ conda create -n py27 python=2.7`

To use Python 2.7: `$ source activate py27`

To check if you have the correct version: `$ python --version`


#### Installing Python 3.5
`$ conda create -n py35 python=3.5`

To use Python 3.5: `source activate py35`

To check if you have the correct version: `$ python --version`