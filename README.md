# Global Tropical Storm Model

This repository contains the work for a project take this
[Typhoon impact model](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model)
developed by 510 Global for usage in the Philippines,
and extend it to be applicable globally.

## Usage

To use this repository, download or sync the `data` directory from
[this URL](https://drive.google.com/drive/folders/15e5BPkhECGeKTObdJIuixICMqhPhVyPK?usp=sharing)

Move the directory to a location that suits you,
and create an environment variable called `STORM_DATA_DIR` that points
to the `data` directory. If for some reason using an environment
variable doesn't work for you, then you'll need to replace each instance
of `os.getenv("STORM_DATA_DIR")` with a string denoting the full path to the
`data` directory.

To execute the notebooks, an installation of Python >=3.8 is required.

If using `venv`, activate your environment and
install the requirements with `pip`:

```shell
pip install -r requirements.txt
```

If using `conda`, create your environment with the `environment.yml` file:

```shell
conda env create --file environment.yml
```

## Development

All code is formatted according to black and flake8 guidelines.
The repo is set-up to use pre-commit.
Before you start developing in this repository, you will need to run

```shell
pre-commit install
```

You can run all hooks against all your files using

```shell
pre-commit run --all-files
```

It is also recommended to use `jupytext`
to convert all Jupyter notebooks (`.ipynb`) to Markdown files (`.md`),
and committing both to version control.

If any additional packages are required, please add them alphabetically
to both `requirements.txt`
(with the version number pinned) and `requirements-conda.txt` (with no version number).
