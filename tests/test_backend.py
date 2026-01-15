"""Tests for SAL xarray backend."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from sal_xarray.main import SALBackendEntrypoint


@pytest.fixture
def mock_signal():
    """Create a mock SAL signal object."""
    signal = Mock()

    # Mock dimension
    dim = Mock()
    dim.data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    dim.temporal = True
    dim.description = "Time dimension"
    dim.units = "s"

    signal.dimensions = [dim]
    signal.data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    signal.description = "Test signal"
    signal.units = "V"

    # Mock mask
    mask = Mock()
    mask.status = np.array([0, 0, 1, 0, 0])
    mask.description = "Data quality mask"
    mask.key = np.array(["good", "bad"])
    signal.mask = mask

    return signal


@pytest.fixture
def mock_sal_client(mock_signal):
    """Create a mock SAL client."""
    with patch("sal_xarray.main.SALClient") as MockClient:
        client_instance = Mock()
        client_instance.get.return_value = mock_signal
        MockClient.return_value = client_instance
        yield MockClient


class TestSALBackendEntrypoint:
    """Tests for SALBackendEntrypoint."""

    def test_guess_can_open_valid(self):
        """Test that guess_can_open returns True for sal:// URLs."""
        backend = SALBackendEntrypoint()
        assert backend.guess_can_open("sal://pulse/12345/signal_name")

    def test_guess_can_open_invalid(self):
        """Test that guess_can_open returns False for non-sal URLs."""
        backend = SALBackendEntrypoint()
        assert not backend.guess_can_open("http://example.com")
        assert not backend.guess_can_open("file://data.nc")

    def test_open_dataset_invalid_type(self):
        """Test that open_dataset raises error for non-string input."""
        backend = SALBackendEntrypoint()
        with pytest.raises(
            ValueError, match="SAL backend only supports string filenames"
        ):
            backend.open_dataset(123)

    def test_open_dataset_missing_scheme(self):
        """Test that open_dataset raises error when sal:// scheme is missing."""
        backend = SALBackendEntrypoint()
        with pytest.raises(
            ValueError, match="SAL dataset must start with the sal:// scheme"
        ):
            backend.open_dataset("pulse/12345/signal_name")

    def test_open_dataset_invalid_format(self):
        """Test that open_dataset raises error for invalid URL format."""
        backend = SALBackendEntrypoint()
        with pytest.raises(ValueError, match="Invalid SAL URL format"):
            backend.open_dataset("sal://invalid/format")

    def test_open_dataset_success(self, mock_sal_client, mock_signal):
        """Test successful dataset opening with mocked SAL client."""
        backend = SALBackendEntrypoint()
        ds = backend.open_dataset("sal://pulse/12345/signal_name")

        # Check that dataset was created
        assert isinstance(ds, xr.Dataset)

        # Check that data variable exists
        assert "data" in ds.data_vars
        assert "status" in ds.data_vars

        # Check dimensions
        assert "time" in ds.dims
        assert ds.sizes["time"] == 5

        # Check data values
        np.testing.assert_array_equal(ds["data"].values, mock_signal.data)

        # Check attributes
        assert ds["data"].attrs["description"] == "Test signal"
        assert ds["data"].attrs["units"] == "V"
        assert ds["data"].attrs["shot_id"] == 12345

        # Check status variable
        np.testing.assert_array_equal(ds["status"].values, mock_signal.mask.status)
        assert ds["status"].attrs["description"] == "Data quality mask"

        # Check that SALClient was called correctly
        mock_sal_client.assert_called_once_with(host="https://sal.jetdata.eu")
        mock_sal_client.return_value.get.assert_called_once()

    def test_open_dataset_custom_host(self, mock_sal_client):
        """Test opening dataset with custom host."""
        backend = SALBackendEntrypoint()
        custom_host = "https://custom.sal.server"
        backend.open_dataset("sal://pulse/12345/signal_name", host=custom_host)

        mock_sal_client.assert_called_once_with(host=custom_host)

    def test_open_dataset_with_multiple_dimensions(self, mock_sal_client):
        """Test opening dataset with multiple dimensions."""
        # Create mock signal with two dimensions
        signal = Mock()

        dim0 = Mock()
        dim0.data = np.array([0.0, 1.0, 2.0])
        dim0.temporal = False
        dim0.description = "Spatial dimension"
        dim0.units = "m"

        dim1 = Mock()
        dim1.data = np.array([0.0, 1.0, 2.0, 3.0])
        dim1.temporal = True
        dim1.description = "Time dimension"
        dim1.units = "s"

        signal.dimensions = [dim0, dim1]
        signal.data = np.random.rand(3, 4)
        signal.description = "2D signal"
        signal.units = "A"

        mask = Mock()
        mask.status = np.zeros((3, 4), dtype=int)
        mask.description = "Mask"
        mask.key = np.array(["good", "bad"])
        signal.mask = mask

        mock_sal_client.return_value.get.return_value = signal

        backend = SALBackendEntrypoint()
        ds = backend.open_dataset("sal://pulse/12345/signal_name")

        # Check dimensions
        assert "dim_0" in ds.dims
        assert "time" in ds.dims
        assert ds.sizes["dim_0"] == 3
        assert ds.sizes["time"] == 4

        # Check data shape
        assert ds["data"].shape == (3, 4)

    def test_open_dataset_client_error(self, mock_sal_client):
        """Test that open_dataset handles SAL client errors."""
        mock_sal_client.return_value.get.side_effect = Exception("Connection failed")

        backend = SALBackendEntrypoint()
        with pytest.raises(RuntimeError, match="Failed to retrieve SAL signal"):
            backend.open_dataset("sal://pulse/12345/signal_name")

    def test_open_dataset_different_shot_numbers(self, mock_sal_client):
        """Test opening datasets with different shot numbers."""
        backend = SALBackendEntrypoint()

        # Test with different shot numbers
        for shot_num in [1, 100, 99999]:
            ds = backend.open_dataset(f"sal://pulse/{shot_num}/signal_name")
            assert ds["data"].attrs["shot_id"] == shot_num

    def test_open_datatree(self, mock_sal_client, mock_signal):
        """Test opening a datatree with the SAL backend."""
        backend = SALBackendEntrypoint()
        dt = backend.open_datatree("sal://pulse/12345/signal_name")

        # Check that a DataTree was created
        assert isinstance(dt, xr.DataTree)

        # Check that dataset is at root
        assert "/" in dt.groups
        ds = dt["/"].ds

        # Check dataset contents
        assert isinstance(ds, xr.Dataset)
        assert "data" in ds.data_vars
        assert "status" in ds.data_vars

        # Check dimensions
        assert "time" in ds.dims
        assert ds.sizes["time"] == 5

        # Check data values
        np.testing.assert_array_equal(ds["data"].values, mock_signal.data)

    def test_open_datatree_with_drop_variables(self, mock_sal_client):
        """Test opening a datatree with drop_variables parameter."""
        backend = SALBackendEntrypoint()
        dt = backend.open_datatree(
            "sal://pulse/12345/signal_name", drop_variables="status"
        )

        # Check that DataTree was created
        assert isinstance(dt, xr.DataTree)

        # The drop_variables parameter is passed through but not implemented
        # in the current backend, so this just verifies it doesn't error
        ds = dt["/"].ds
        assert isinstance(ds, xr.Dataset)


class TestXarrayIntegration:
    """Test integration with xarray's open_dataset function."""

    def test_xr_open_dataset_with_engine(self, mock_sal_client):
        """Test using xr.open_dataset with engine='sal'."""
        # This test requires the backend to be registered
        # In practice, this happens via entry_points in pyproject.toml
        ds = xr.open_dataset("sal://pulse/12345/signal_name", engine="sal")

        assert isinstance(ds, xr.Dataset)
        assert "data" in ds.data_vars
        assert "status" in ds.data_vars

    def test_xr_open_datatree_with_engine(self, mock_sal_client):
        """Test using xr.open_datatree with engine='sal'."""
        # This test requires the backend to be registered
        # In practice, this happens via entry_points in pyproject.toml
        dt = xr.open_datatree("sal://pulse/12345/signal_name", engine="sal")

        assert isinstance(dt, xr.DataTree)
        assert "/" in dt.groups
        ds = dt["/"].ds
        assert isinstance(ds, xr.Dataset)
        assert "data" in ds.data_vars
        assert "status" in ds.data_vars
