"""Xarray SAL backend entrypoint."""

import os
import re
from typing import Any, Iterable

import numpy as np
import xarray as xr
from sal.client import SALClient

from xarray.backends import BackendEntrypoint
from xarray.backends.common import ReadBuffer
from xarray.backends.common import AbstractDataStore


class SALBackendEntrypoint(BackendEntrypoint):
    """Xarray SAL backend entrypoint."""

    def open_dataset(
        self,
        filename_or_obj: str
        | os.PathLike[Any]
        | ReadBuffer
        | bytes
        | memoryview
        | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
        host: str = "https://sal.jetdata.eu",
    ) -> xr.Dataset:
        """Open a SAL dataset given a signal name and shot number.

        Parameters
        ----------
        filename_or_obj : str
            SAL dataset specified as sal://pulse/<shot_number>/<signal_name>
        drop_variables : str or list of str, optional
            Variables to drop from the dataset (not used).
        """

        if not isinstance(filename_or_obj, str):
            raise ValueError("SAL backend only supports string filenames.")

        if "sal://" not in filename_or_obj:
            raise ValueError("SAL dataset must start with the sal:// scheme")

        name = filename_or_obj.replace("sal://", "")
        match = re.search(r"sal://pulse/(\d+)/", filename_or_obj)

        if not match:
            raise ValueError(
                "Invalid SAL URL format. Expected sal://pulse/<shot_number>/<signal_name>"
            )

        shot = int(match.group(1))
        name = "/" + name

        client = SALClient(host=host)
        try:
            signal = client.get(name)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve SAL signal '{name}': {e}") from e

        data = self.parse_data(signal)
        data.attrs["shot_id"] = shot

        status = xr.DataArray(signal.mask.status, dims=data.dims, coords=data.coords)
        status.attrs["description"] = signal.mask.description
        status.attrs["flag_values"] = np.unique(signal.mask.status)
        status.attrs["flag_meanings"] = signal.mask.key.tolist()

        dataset = data.to_dataset(name="data")
        dataset["status"] = status
        return dataset

    def parse_data(self, signal) -> xr.DataArray:
        """Parse a SAL LeafReport into an xarray DataArray."""

        dims = {}
        for index, item in enumerate(signal.dimensions):
            name = f"dim_{index}"
            if item.temporal:
                name = "time"
            dim = xr.DataArray(item.data, dims=[name], name=name)
            dim.attrs["description"] = item.description
            dim.attrs["units"] = item.units

            dims[dim.name] = dim

        data = xr.DataArray(
            signal.data,
            dims=[d.name for d in dims.values()],
            coords={d.name: d for d in dims.values()},
        )

        data.attrs["description"] = signal.description
        data.attrs["units"] = signal.units
        data.attrs["sal_name"] = name

        return data

    def open_datatree(self, filename_or_obj, *, drop_variables=None):
        dataset = self.open_dataset(filename_or_obj, drop_variables=drop_variables)
        return xr.DataTree(dataset)

    def open_groups_as_dict(self, filename_or_obj, **kwargs):
        """Open groups as a dictionary (not supported for SAL backend)."""
        raise NotImplementedError("SAL backend does not support opening groups")

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj):
        return filename_or_obj.startswith("sal://")

    description = "Use SAL data in Xarray"

    url = "https://github.com/samueljackson92/sal-xarray"
