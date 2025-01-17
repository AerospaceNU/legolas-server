# legolas-server


## About
This is the server repository for the LEGOLAS IRAD project. The server runs on the NVIDIA Jetson compute module connected to the tracking gimbal and camera system.

If you are trying to set up your laptop to be able to run LEGOLAS, you should be in the [AerospaceNU/legolas-gui](https://github.com/AerospaceNU/legolas-gui) repository.

## Installation

### For Users
TODO 

(Ideally we are building an executable and directing users to the releases page to download it).

### For Developers
This repository uses Python 3.12 and [pipenv](https://pipenv.pypa.io/en/latest/) to manage Python versioning and dependencies. Package versions are specified in the `Pipfile` and `Pipfile.lock`. 

- First, make sure you have Python 3.12 installed. 

    - You can download Python 3.12 from the [Python.org downloads page](https://www.python.org/downloads/)

    - Alternatively:

        - On Linux/MacOS, you can also use [pyenv](https://github.com/pyenv/pyenv), which is great for managing multiple Python versions.
        - On Windows, I have found that downloading from the Microsoft Store works quite well.
- Next, run `python --version` to verify your version. (You may need to use `python3` or `python312` or some other method to get the right executable).
    - Your output should look like
        ```bash
        you@computer % python --version
        Python 3.12.7
        ```
    
- **Using the same Python executable that got you the version output above**, run 
    ```bash 
    python -m pip install setuptools wheel pipenv
    ```
- Now, run 
    ```bash 
    pipenv install --dev
    pipenv shell
    ```
    to install all dependencies and activate the environment. 
- Your shell should now look something like
    ```bash
    (legolas-server) sam@Enceladus legolas-server %
    ```
    and running 
    ```bash
    (legolas-server) sam@Enceladus legolas- % which python
    /Users/sam/.local/share/virtualenvs/legolas-server-Pu7jE4Of/bin/python
    ```
    should look something like this.

## Running
### Server
In the above `pipenv` activated environment, run

```bash
% python src/main.py
```
to run the server.

### Tests
To run the test suite, run
```bash
% pytest test
```

### Formatting & Linting
To run the formatting, import sorting, linting, etc.
```bash
% black src test
% isort src test
% mypy src test
% flake8 src test
```

## VSCode Setup
I recommend installing the following extensions if you plan to use VSCode
https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter
https://marketplace.visualstudio.com/items?itemName=ms-python.isort
https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance
https://marketplace.visualstudio.com/items?itemName=ms-python.python
https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy
