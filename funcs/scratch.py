import xarray as xr
import logging
import warnings
import getpass
import numpy as np
import pandas as pd
import argopy
import gsw
from argopy.fetchers import DataFetcher as ArgoDataFetcher
from argopy.fetchers import GDACFetcher as GDACArgoDataFetcher

class CustomGDACArgoDataFetcher(GDACArgoDataFetcher):
    def to_xarray(self, errors: str = "ignore"):
        """Load Argo data and return a :class:`xarray.Dataset`

        Parameters
        ----------
        errors: str, default='ignore'
            Define how to handle errors raised during data URIs fetching:

                - 'ignore' (default): Do not stop processing, simply issue a debug message in logging console
                - 'silent':  Do not stop processing and do not issue log message
                - 'raise': Raise any error encountered

        Returns
        -------
        :class:`xarray.Dataset`
        """
        if len(self.uri) == 0:
            raise DataNotFound("No data found for: %s" % self.indexfs.cname)

        # Load datasets and filter to include only those that contain the 'PSAL' variable
        valid_datasets = []
        for uri in self.uri:
            try:
                temp_ds = self.fs.open_dataset(uri)
                if 'PSAL' in temp_ds.variables:
                    valid_datasets.append(temp_ds)
                else:
                    logging.debug(f"Dataset {uri} does not contain the 'PSAL' variable.")
            except Exception as e:
                if errors == 'raise':
                    raise e
                elif errors == 'ignore':
                    logging.debug(f"Error loading {uri}: {e}")

        if len(valid_datasets) == 0:
            raise DataNotFound("No valid data found containing the 'PSAL' variable.")

        # Align datasets along the N_PROF dimension
        max_prof_size = max(ds.sizes['N_PROF'] for ds in valid_datasets)
        for i, ds in enumerate(valid_datasets):
            if ds.sizes['N_PROF'] < max_prof_size:
                valid_datasets[i] = ds.reindex({'N_PROF': range(max_prof_size)})

        # Concatenate valid datasets
        ds = xr.concat(valid_datasets, dim="N_POINTS")

        # Meta-data processing:
        ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))  # Re-index to avoid duplicate values
        ds = ds.set_coords("N_POINTS")
        ds = ds.sortby("TIME")

        # Remove netcdf file attributes and replace them with simplified argopy ones:
        if "Fetched_from" not in ds.attrs:
            raw_attrs = ds.attrs
            ds.attrs = {}
            ds.attrs.update({"raw_attrs": raw_attrs})
            if self.dataset_id == "phy":
                ds.attrs["DATA_ID"] = "ARGO"
            if self.dataset_id in ["bgc", "bgc-s"]:
                ds.attrs["DATA_ID"] = "ARGO-BGC"
            ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
            ds.attrs["Fetched_from"] = self.server
            try:
                ds.attrs["Fetched_by"] = getpass.getuser()
            except:
                ds.attrs["Fetched_by"] = "anonymous"
            ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime(
                "%Y/%m/%d"
            )

        ds.attrs["Fetched_constraints"] = self.cname()
        if len(self.uri) == 1:
            ds.attrs["Fetched_uri"] = self.uri[0]
        else:
            ds.attrs["Fetched_uri"] = ";".join(self.uri)

        return ds

# Create an instance of CustomGDACArgoDataFetcher
argo_loader = CustomGDACArgoDataFetcher(src="gdac", ftp="/swot/SUM05/dbalwada/Argo_sync", progress=True)

def get_box(box, interp_step=2):
    """Takes latitude/longitude/depth data and a sample rate and returns an xarray with CT, SA, SIG0, and SPICE interpolated to a pressure grid of 2m.

    box: lat/lon in the form: box=[lon_min, lon_max, lat_min, lat_max, depth_min, depth_max]
    sample_min: minimum sample rate [m]
    """

    ds = argo_loader.region(box)
    #print("loading points complete")

    ds = ds.to_xarray()
    print("to xarray complete")

    #ds = ds.argo.teos10(["CT", "SA", "SIG0"])
    ds = ds.argo.point2profile()
    print("point to profile complete")

    ds_interp = get_ds_interp(ds, box[4], box[5], interp_step)
    print("interpolation complete")

    ds_interp["SPICE"] = gsw.spiciness0(ds_interp.SA, ds_interp.CT).rename("SPICE")
    print("adding spice complete")

    ds_interp = get_MLD(ds_interp)
    ds_interp = add_times(ds_interp)
    print("adding MLD complete")
    
    if 'raw_attrs' in ds_interp.attrs:
        del ds_interp.attrs['raw_attrs']

    return ds_interp
