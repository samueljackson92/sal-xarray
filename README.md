# sal-xarray

An xarray backend for SAL (Simple Access Layer) that enables seamless integration of SAL data sources with the xarray ecosystem.

## Overview

sal-xarray provides a backend plugin for xarray that allows you to access SAL data sources using xarray's familiar API. It automatically handles the conversion of SAL signals to xarray DataArrays and Datasets, making it easy to work with SAL data in scientific Python workflows.
## Features

- **xarray Integration**: Access SAL data using xarray's `open_dataset` function
- **Automatic Conversion**: Converts SAL signals to xarray DataArrays with proper coordinates and metadata
- **Error Handling**: Includes error data alongside signal data
- **URI-based Access**: Simple URI scheme (`sal://pulse/<shot_number>/<signal_name>`) for accessing data

## Installation 

```bash
pip install sal-xarray
```

Or using `uv`:

```bash
uv pip install sal-xarray
```

## Usage

Open a SAL dataset using xarray:

```python
import xarray as xr

# Open a SAL signal by name and shot number
ds = xr.open_dataset("sal://pulse/87737/ppf/signal/jetppf/magn/ipla", engine="sal")

# Access the data
data = ds["data"]
status = ds["status"]

# The dataset includes time coordinates
time = ds["time"]

# Access metadata
units = ds["data"].attrs["units"]
signal_name = ds["data"].attrs["sal_name"]
```

The URI format is: `sal://pulse/<shot_number>/<signal_name>`

## Data Structure

When you open a SAL dataset, sal-xarray creates an xarray Dataset with:
- **data**: The signal data as a DataArray
- **status**: The status data as a DataArray
- **time**: Time coordinates (dimension)
- **attrs**: Metadata including units and SAL signal name

## Development

### Setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/samueljackson92/sal-xarray.git
cd sal-xarray
uv sync
```

### Running Tests

```bash
uv run pytest tests/
```

### Code Quality

The project uses ruff for linting and formatting, and pylint for additional checks:

```bash
# Run ruff
uv run ruff check sal_xarray tests
uv run ruff format sal_xarray tests

# Run pylint
uv run pylint sal_xarray

# Run ty
uv run ty check sal_xarray
```

## Contributing

Contributions are welcome! Please ensure:

1. Tests pass for any new functionality
2. Code follows the project's style guidelines (ruff and pylint)
3. Documentation is updated as needed

## License

MIT License
