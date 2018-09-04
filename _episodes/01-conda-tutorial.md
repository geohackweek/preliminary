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
[**Conda**](http://conda.pydata.org/docs/) is an **open source `package` and `environment` management system for any programming languages, but very popular among python community,** for installing multiple versions of software packages, their dependencies and switching easily between them. It works on Linux, OS X and Windows.

### Installing Miniconda

##### Windows
Click [here](http://conda.pydata.org/miniconda.html) to download the proper installer for your Windows platform (64 bits).
We recommend to download the Python 3 version of Miniconda. You can still create Python 2 environments using the Python 3 version of Miniconda.

When installing, you will be asked if you wish to make the Anaconda Python your default Python for Windows.
If you do not have any other installation that is a good option. If you want to keep multiple versions of python on your machine (e.g. ESRI-supplied python, or 64 bit versions of Anaconda), then don't select the option to modify your path or modify your windows registry settings.

##### Linux and OSX
You may follow manual steps from [here](http://conda.pydata.org/miniconda.html) similar to the instructions on Windows (see above). Alternatively, you can execute these commands on a terminal shell (in this case, the bash shell):

```bash
url=bit.ly/miniconda3
wget $url -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda update conda --yes
```

### Installing Anaconda (Optional)

*NOTE: If you don't have time or disk space for the entire distribution do not install Anaconda. Install only [Miniconda](http://conda.pydata.org/miniconda.html), a bootstrap version of Anaconda, which contains only Python, essential packages, and conda. We will provide an environment file to install all the packages necessary for the Oceanhackweek.*

[Anaconda](https://www.anaconda.com/distribution/) is a data science platform that comes with a lot of packages. At the core, Anaconda uses conda package management system.

List of packages included can be found [*here*](https://docs.anaconda.com/anaconda/packages/pkg-docs)

1. To install Anaconda, please click on the link below for your operating system, and follow the instructions on the [site](https://www.anaconda.com/download/).
2. Once Anaconda installation step is finished run `python` in the command line to test if Anaconda is installed correctly. **Note: For windows, please use Anaconda prompt as the command line. It should be installed with your installation of Anaconda**
If Anaconda is installed correctly, you should have this prompt, which emphasizes **Anaconda**:

```bash
$ python
Python 3.6.4|Anaconda custom (x86_64)| (default, Jan 16 2018, 18:57:58)
...
```

> ## Additional steps for Windows
> If you did not add conda to the PATH environment variable during the installation, please follow the steps below to manually add conda to your PATH environment variable.
> 1. Open up **Anaconda Prompt** and figure out the location of `conda` and `python`.
```
where conda
where python
```
You should get the location of `conda` (`C:\Users\lsetiawan\Miniconda3\Scripts\conda.exe`) and `python` (`C:\Users\lsetiawan\Miniconda3\python.exe`)
> 2. Open up **Windows Command Prompt** or **Windows Powershell** and use the `SETX` command to add conda to path
```
SETX PATH "%PATH%;C:\Users\lsetiawan\Miniconda3\Scripts;C:\Users\lsetiawan\Miniconda3"
```
{: .callout}

### Installing Python
We will be using Python 3.6 during the week. Each tutorial will have their own conda environment that contains the correct libraries to carry out the tutorial. If you are already familiar with Python 2.7, you can take a look at the syntax differences [here](http://sebastianraschka.com/Articles/2014_python_2_3_key_diff.html), but the main point to remember is to put the print statements in parentheses:
```python
print('Hello World!')
```


``` bash
$ conda create -n py36 python=3.6
```

To use Python 3.6: 

``` bash
$ source activate py36
```

To check if you have the correct version: 

``` bash
$ python --version
```
