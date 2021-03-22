
# Pre-requisite
* Python 3.9.x installed

# Check you environment variable
At first, you need to find your python path (ex. `C:/Users/[yourname]/AppData/Local/Programs/Python39`).

For both your Local and System environment, create a new variable ([How do I set or change the PATH system variable?](https://java.com/en/download/help/path.html)): `PATH`

In the "New User Variable" windows opened:
* Variable Name : `PATH` (respect the case)
* Variable value: `C:/Users/[yourname]/AppData/Local/Programs/Python39;C:/Users/[yourname]/AppData/Local/Programs/Python39/Scripts` (allow you to use this version of Python and Pip Install scripts)

Click on the OK button for all the windows.

# Configure your local Python
* Go into your Python folder (ex. `C:/Users/[yourname]/AppData/Local/Programs/Python39`).
* Make a copy of the `python.exe` file, and change the name for `python3.exe`.

Now you must have in your folder `python.exe` and `python3.exe`

# Check the python version
Open `GitBash` (or equivalent software) and check the python version used:

```bash
python --version
# Python 3.9.2

```

```bash
python3 --version
# Python 3.9.2
```

# Troubleshoot
If you have some problems with the wheel like `The wheel package is not available`:

```bash
# try to update your pip version
python -m pip install --upgrade pip

# try to install/update the package with the error (ex. For `The wheel package is not available` we try to install the `wheel` package)
pip install wheel

```