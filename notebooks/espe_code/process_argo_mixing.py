#!/usr/bin/env python3
"""Process local Argo profiles and all NATRE scientific analysis in one script.

Use the existing Argo_Feb_25 environment (editable Argopy and neutral_density).
All datasets, manifests, logs and Dask spill must be under /swot/SUM05/amf2288.
Code lives in the repository; Python/Numba/Matplotlib caches default to /tmp.

First run the mandatory 2014-2023 NATRE comparison (end date is exclusive)::

    python -B process_argo_mixing.py validate-natre \
        --output /swot/SUM05/amf2288/argo_mixing/natre_validation \
        --workers 8 --batch-size 64

Inspect local coverage without processing profiles::

    python -B process_argo_mixing.py inventory --bounds -38 -20 19 32 \
        --start 2014-01-01 --end 2024-01-01 \
        --output /swot/SUM05/amf2288/argo_mixing/inventory

Run a selected region, including all available dates if start/end are omitted::

    python -B process_argo_mixing.py run --bounds -38 -20 19 32 \
        --start 2014-01-01 --end 2024-01-01 --cell-size 0 \
        --output /swot/SUM05/amf2288/argo_mixing/natre

Global/large requests require --validation-report pointing to a passing NATRE
report for this script and archive. No automatic global launch occurs. A mismatch
returns exit code 2 and preserves the comparison report. Validation separates:
(1) replay of saved research-mode raw profiles; (2) local GDAC end-to-end results;
(3) deterministic historical summary fields; (4) independently executed original
notebook science with the same bootstrap seed. Historical unseeded bootstrap
realizations are reported separately and never labelled numerically identical.
If local profile equivalence already fails, the report additionally checks all
reference profiles for deterministic summary fields and a 100-profile subset
with full seeded bootstraps; it never treats these diagnostics as a passing gate.

Output directory: profiles/<cell>/*.nc, summaries/<cell>.nc, per-batch status JSON,
manifest.json and inventory.json. Profile NetCDFs retain Z/time and Zfs/time;
summaries retain SG/TIME plus geographic bounds and coverage counts. Open profile
partitions with xr.open_mfdataset(paths, combine='nested', concat_dim='time').
Restart with identical settings to reuse successfully completed profile batches.
Summaries are recomputed. Workers write their own files and return small statuses.

The initial scientific version deliberately preserves the legacy formulas,
including documented known quirks, to make the required comparison meaningful.
Research QC, <=5 dbar mean spacing above 1000 dbar, 2 m averaging to 2000 m,
and 200 m Mixsea windows remain in force. All indexed profiles are considered;
not all pass scientific eligibility. The default density range and 4 m vertical
correlation scale are the NATRE assumptions: configure them explicitly for other
regions. Unsupported metrics carry flags/NaNs rather than invented values.
"""

import argparse
import collections
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import warnings

# Do this before scientific imports, including in spawned Dask workers. Never
# write import caches beside the user's editable packages in the home directory.
sys.dont_write_bytecode = True
_runtime_base = Path(os.environ.get('ARGO_RUNTIME_DIR', '/tmp/argo-mixing-runtime')).resolve()
if not (_runtime_base.is_relative_to(Path('/tmp').resolve())
        or _runtime_base.is_relative_to(Path('/swot/SUM05/amf2288').resolve())):
    raise ValueError('ARGO_RUNTIME_DIR must be below /tmp or /swot/SUM05/amf2288')
_runtime_base.mkdir(parents=True, exist_ok=True)
for _name, _subdir in [('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib'),
                       ('XDG_CACHE_HOME', 'cache')]:
    os.environ[_name] = str(_runtime_base / _subdir)
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_name] = '1'

"""Legacy-compatible per-profile Argo mixing calculations.

The default scientific formulas deliberately reproduce runs_all_years_gamman.py.
Known legacy conventions are recorded in output provenance.  Input data must
already have the same argopy research-mode QC and adjusted-variable selection.
No downloading, saving, plotting, or Dask scheduling is performed here.
"""
import json
import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr
import gsw
import mixsea as mx
import neutral_density as nds


def _profile_option(config, name, default):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def first_centered_differences(z, values):
    """Centered differences used only by the optional MLD gradient criterion."""
    z, values = np.asarray(z), np.asarray(values)
    result = np.full(values.shape, np.nan, dtype=float)
    if result.size >= 3:
        result[1:-1] = (values[2:] - values[:-2]) / (z[2:] - z[:-2])
        result[0], result[-1] = result[1], result[-2]
    return result


def mixed_layer_depth(z0, den0, Dd = 0.03, crit = "diff", z_min = 30., intrp = True):
    """Mixed layer depth; default difference criterion follows legacy code."""
    if crit != "diff" and crit != "grad" and crit != "DO":
        crit = "diff"
        print("Incorrect criterion, set to diff")
    c,f = den0.shape
    MLD = np.full(f, np.nan)
    for i in range(f):
        if z0.ndim ==1:
            z = np.copy(z0)
        else:
            z = z0[:,i]
        #den = np.sort(den0[:,i])
        den = den0[:,i]
        iif = np.isfinite(den+z)
        if np.sum(iif)<=1:
            continue
        den = den[iif]
        z = z[iif]
        if np.min(z0)>z_min:
            continue

        if crit == "diff":
            sden = den[0]
            denp = den-sden
            imld = np.where( denp>=Dd )[0]
            if imld.size == 0:
                MLD[i] = np.max(z)
            elif imld[0]>0:
                imld = imld[0]
                z2 = z[imld]
                z1 = z[imld-1]
                denp2 = denp[imld]
                denp1 = denp[imld-1]
                if intrp:
                    MLD[i] = (z2-z1)/(denp2-denp1)*(Dd - denp1) + z1
                else:
                    MLD[i] = (z1+z2)*0.5
            else:
                MLD[i] = np.max(z)
                #MLD[i] = z0[0,i]
                
        elif crit == "grad":
            grden = np.abs(first_centered_differences(z, den))
            imld = np.where(grden>=Dd)[0]
            if imld.size == 0:
                MLD[i] = np.max(z)
            elif imld[0]>0:
                imld = imld[0]
                z2 = z[imld]
                z1 = z[imld-1]
                grd2 = grden[imld]
                grd1 = grden[imld-1]
                if intrp:
                    MLD[i] = (z2-z1)/(grd2-grd1)*(Dd - grd1) + z1
                else:
                    MLD[i] = 0.5*(z1+z2)
            else:
                MLD[i] = z[0]

        if crit == "DO":
            sden = den[0]
            denp = den-sden
            imld = np.where( np.abs(denp)>=Dd )[0]
            if imld.size == 0:
                MLD[i] = np.max(z)
            elif imld[0]>0:
                imld = imld[0]
                z2 = z[imld]
                z1 = z[imld-1]
                MLD[i] = z1
            else:
                MLD[i] = np.max(z)
                #MLD[i] = z0[0,i]
                
    return MLD

#this is just a useful function        
def moving_average(x,n, window = "flat"):
    if n%2 == 0:
        n+=1
    N = x.size
    cx = np.full(x.size, np.nan)
    for i in range(N):
        ii = np.arange(i-n//2, i+n//2+1,1)
        if window == "flat":
            ww = np.ones(ii.size)
        elif window == "gauss":
            xx = ii - i
            
            ww = np.exp(- xx**2/(float(n)/4)**2 )
        elif window == "hanning":
            ww = np.hanning(ii.size)
        ww = ww[ (ii>=0) & (ii<N)]
        ii = ii[ (ii>=0) & (ii<N)]
        
        #print(ii)
        kk = np.isfinite(x[ii])
        if np.sum(kk)<0.25*ii.size:
            continue
        cx[i] = np.sum(x[ii[kk]]*ww[kk])/np.sum(ww[kk])
    return cx

def _profile_identity_values(ds, names, indices, default=""):
    """Read an optional one-value-per-profile field without losing strings."""
    for name in names:
        if name not in ds or "N_PROF" not in ds[name].dims:
            continue
        value = ds[name].isel(N_PROF=indices).transpose("N_PROF", ...).values
        value = np.asarray(value)
        if value.ndim > 1:
            if value.dtype.kind not in "SU":
                continue
            value = np.array(["".join(np.asarray(row).astype(str).ravel()) for row in value])
        return np.char.strip(value.astype(str))
    return np.array([str(default)] * len(indices), dtype=str)


def process_profiles(ds, config=None):
    """Process one bounded batch of already-QC'd Argo profiles.

    Parameters
    ----------
    ds : xarray.Dataset
        Variables PRES, TEMP, PSAL on (N_PROF, N_LEVELS), and TIME,
        LONGITUDE, LATITUDE on N_PROF.  Optional platform/cycle/direction,
        PROFILE_ID and SOURCE_FILE fields are carried to the result.
    config : mapping or namespace, optional
        dz (2 m), max_depth (2000 m), min_mean_res_in_upper_1000m (5 dbar),
        min_resolution_micro (10 m).  Default values match legacy processing.

    Returns
    -------
    xarray.Dataset
        Legacy variables on (Z, time)/(Zfs, time), with identity and diagnostic
        status fields on time.  Only profiles passing the legacy pressure
        resolution filter are included.  Batch accounting is in attrs.
        Unsupported gamma_n latitudes and explicit library failure sentinels
        produce missing gamma-dependent metrics, with reason flags. Unexpected
        exceptions propagate so infrastructure/programming errors are visible.
    """
    required = ("PRES", "TEMP", "PSAL", "TIME", "LONGITUDE", "LATITUDE")
    missing = [name for name in required if name not in ds]
    if missing:
        raise ValueError("Missing required profile variables: " + ", ".join(missing))
    if "N_PROF" not in ds.dims or "N_LEVELS" not in ds.dims:
        raise ValueError("Expected N_PROF and N_LEVELS dimensions")
    DS = ds.copy(deep=False)
    for name in ("PRES", "TEMP", "PSAL"):
        DS[name] = DS[name].transpose("N_PROF", "N_LEVELS")
    # The caller bounds batch size; materialize this batch once, not repeatedly.
    DS = DS.load()
    dz = float(_profile_option(config, "dz", 2.0))
    max_depth = float(_profile_option(config, "max_depth", 2000.0))
    min_mean_res_in_upper_1000m = float(_profile_option(config, "min_mean_res_in_upper_1000m", 5.0))
    min_resolution_micro = float(_profile_option(config, "min_resolution_micro", 10.0))
    if dz <= 0 or max_depth < 2 or min_mean_res_in_upper_1000m <= 0 or min_resolution_micro <= 0:
        raise ValueError("Depth/resolution configuration must be positive")
    # These empty-bin warnings are inherent to legacy 2 m averaging.  Library
    # exceptions and other warnings are deliberately not swallowed.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="divide by zero encountered in", category=RuntimeWarning)
        return _process_profiles_legacy(DS, dz, max_depth, min_mean_res_in_upper_1000m, min_resolution_micro)


def _process_profiles_legacy(DS, dz, max_depth, min_mean_res_in_upper_1000m, min_resolution_micro):
    Z = np.arange(2,max_depth+dz,dz)
    
    ##grids data
    NP = Z.size
    N_PROF0 = DS.N_PROF.compute().size


    p00 = DS.PRES.values.T
    dpdz = np.abs(np.diff(p00, axis = 0))
    mdpdz = np.full(N_PROF0, np.nan)
    for i in range(N_PROF0):
        dpdz00 = dpdz[p00[:-1,i]<1000,i]
        mdpdz[i] = np.nanmean(dpdz00)

    iigood = np.where(mdpdz<=min_mean_res_in_upper_1000m)[0]
    N_PROF = iigood.size


    Time = np.array(DS.TIME.compute())[iigood]

    Lon = np.array(DS.LONGITUDE.compute())[iigood]
    Lat = np.array(DS.LATITUDE.compute())[iigood]
    position_valid = np.isfinite(Lon) & np.isfinite(Lat) & (np.abs(Lat) <= 90) & (Lon >= -180) & (Lon <= 360)
    gamma_status = np.zeros(N_PROF, dtype=np.int16)
    gamma_failed_count = np.zeros(N_PROF, dtype=np.int32)
    mixsea_status = np.zeros(N_PROF, dtype=np.int16)
    polynomial_status = np.zeros(N_PROF, dtype=np.int16)
    n2_status = np.zeros(N_PROF, dtype=np.int16)
    stage_message = np.full(N_PROF, "", dtype=object)

    TEMP = np.full( (NP, N_PROF), np.nan )
    PSAL = np.full( (NP, N_PROF), np.nan )

    CT = np.full( (NP, N_PROF), np.nan )
    SA = np.full( (NP, N_PROF), np.nan )

    dCTdz = np.full( (NP, N_PROF), np.nan )
    dSAdz = np.full( (NP, N_PROF), np.nan )

    SG1 = np.full( (NP, N_PROF), np.nan )
    SG0 = np.full( (NP, N_PROF), np.nan )

    P = np.full( (NP, N_PROF), np.nan )
    Zm = np.full( (NP, N_PROF), np.nan )
    for i,j in enumerate(iigood):
        if not position_valid[i]:
            continue

        p0= np.array(DS.isel(N_PROF=j).PRES.compute())
        t0 = np.array(DS.isel(N_PROF=j).TEMP.compute())
        s0 = np.array(DS.isel(N_PROF=j).PSAL.compute())
        ii = np.isfinite(t0+s0)

        t0 = t0[ii]
        s0 = s0[ii]
        p0 = p0[ii]

        sa0 = gsw.SA_from_SP(s0, p0, Lon[i], Lat[i]);
        # LEGACY COMPATIBILITY: s0 is practical salinity here, as in the source.
        # Correcting this to sa0 would change scientific results and is deferred.
        ct0 = gsw.CT_from_t(s0, t0, p0);
        depth0 = np.abs(gsw.z_from_p(p0,Lat[i]))
        for k in range(Z.size):
            kk = np.where(np.abs(depth0-Z[k])<=dz/2)[0]
            if kk.size==0:
                continue

            TEMP[k,i] = np.mean(t0[kk])
            PSAL[k,i] = np.mean(s0[kk])
            P[k,i] = np.mean(p0[kk])
            Zm[k,i] = np.mean(depth0[kk])
            CT[k,i] = np.mean(ct0[kk])
            SA[k,i] = np.mean(sa0[kk])

            kktop = np.where(np.abs(depth0-(Z[k]-dz/2))<=dz/2)[0]
            kkbottom = np.where(np.abs(depth0-(Z[k]+dz/2))<=dz/2)[0]

            dCTdz[k, i] = (np.nanmean(ct0[kktop]) - np.nanmean(ct0[kkbottom])) / (np.nanmean(depth0[kktop]) - np.nanmean(depth0[kkbottom]))
            dSAdz[k, i] = (np.nanmean(sa0[kktop]) - np.nanmean(sa0[kkbottom])) / (np.nanmean(depth0[kktop]) - np.nanmean(depth0[kkbottom]))

    SG0 = gsw.sigma0(SA, CT)
    SG1 = gsw.sigma1(SA,CT)
    Spi0 = gsw.spiciness0(SA,CT)
    Spi1 = gsw.spiciness1(SA,CT)


    ##gamman

    gamman = np.full((NP, N_PROF), np.nan)
    for j in range(N_PROF):
        if not position_valid[j]:
            gamma_status[j] = 2
            continue
        # Bounds documented by the installed neutral_density.gamma_n function.
        if Lat[j] < -80 or Lat[j] > 64:
            gamma_status[j] = 1
            continue
        ii = np.isfinite(PSAL[:,j])
        if np.sum(ii) == 0:
            gamma_status[j] = 3
            continue
        gamma_values, _, _ = nds.gamma_n(PSAL[ii,j],TEMP[ii,j],P[ii,j],Lon[j],Lat[j])
        gamma_values = np.asarray(gamma_values, dtype=float)
        failed = ~np.isfinite(gamma_values) | (gamma_values <= 0)
        gamma_failed_count[j] = np.count_nonzero(failed)
        if np.any(failed):
            gamma_status[j] = 4
            gamma_values = gamma_values.copy()
            gamma_values[failed] = np.nan
        gamman[ii,j] = gamma_values


    gamman = np.array(gamman)
    N2 = np.full(gamman.shape, np.nan)
    for i in range(N_PROF):

        gamman0 = np.copy(gamman[:,i])
        ii = np.isfinite(gamman0)
        z00 = Zm[ii,i]
        gamman00 = gamman0[ii]
        gamman00 = np.sort(gamman00)
        if gamman00.size < 2:
            n2_status[i] = 1
            continue
        N20 = np.full(gamman00.size, np.nan)
        N20[1:-1] = 9.81/(1000+gamman00[1:-1])*(gamman00[2:]-gamman00[:-2])/(z00[2:]-z00[:-2])

        N20[0] = N20[1]
        N20[-1] = N20[-2]
        N2[ii,i] = N20
    
    
    ##mixing

    # Center points of depth windows. Windows are half overlapping, i.e.
    # The legacy code uses 200 m windows and 50 m center spacing (75% overlap).
    window_size = 200
    min_size = 10.
    dZ = window_size // 4
    #print("window size {} m, window spacing {} m".format(window_size, dz))
    depth_bin = np.arange(dZ, int(max_depth) if max_depth.is_integer() else max_depth, dZ)
    # Wavenumber vector. Starts at wavenumber corresponding to a 200m wavelength.
    m = np.arange(2 * np.pi / window_size, 2 * np.pi / min_size, 2 * np.pi / window_size)
    # Wavenumber indices for integration. Shear is integrated from 300m to
    # 100m scales. Strain is integrated from 150m to 30m.
    m_include_sh = list(range(3))
    m_include_st = list(range(1, 12))
    
    eps_st = np.full( (depth_bin.size, N_PROF), np.nan)
    Krho_st = np.full( (depth_bin.size, N_PROF), np.nan)
    depth_st = np.full( (depth_bin.size, N_PROF), np.nan)
    for i in range(N_PROF):
        #print(i)
        if not position_valid[i]:
            mixsea_status[i] = 4
            continue
        if Lat[i] == 0:
            mixsea_status[i] = 2
            continue
        if np.sum(np.isfinite(TEMP[:,i]+PSAL[:,i]))<100:
            mixsea_status[i] = 1
            continue
        try:
            eps_shst, krho_shst, diag = mx.shearstrain.nan_shearstrain(
                Z,
                TEMP[:,i],
                PSAL[:,i],
                Lon[i],
                Lat[i],
                TEMP[:,i]*0.,
                TEMP[:,i]*0.,
                Z,
                m=m,
                depth_bin=depth_bin,
                window_size=window_size,
                m_include_sh=m_include_sh,
                m_include_st=m_include_st,
                ladcp_is_shear=True,
                smooth="AL",
                return_diagnostics=True,
            )
        except ValueError as e:
            mixsea_status[i] = 3
            stage_message[i] = "mixsea ValueError: " + str(e)
            continue
            
        if not np.any(np.isfinite(diag['eps_st'])):
            mixsea_status[i] = 5
        z0 = diag['depth_bin']
        for j in range(z0.size):
            jj = np.where(depth_bin == z0[j])[0]
            if jj.size == 0:
                continue
            eps_st[jj[0],i] =  diag['eps_st'][j]
            Krho_st[jj[0],i] = diag['krho_st'][j]
            
              
    #polynom

    h = 400 #defines the segment length
    POLY_ORDER = 2
    CT_mean = np.full(CT.shape, np.nan)
    SA_mean = np.full(CT.shape, np.nan)
    OXY_mean = np.full(CT.shape, np.nan)
    for i in range(Lon.size):
        #print(i)
        pres0 = P[:,i]
        sg10 = gamman[:,i]
        iif = np.isfinite(sg10)
        if not np.any(iif):
            polynomial_status[i] = 1
            continue
        sg1s = np.full(sg10.size, np.nan)
        sg1s[iif] = np.sort(sg10[iif])#moving_average(np.sort(sg10[iif]),10)
        #sg_smooth = moving_average(gamman_s[:,i],10)
        for j in range(pres0.size):
            deepest = np.where(np.isfinite(CT[:,i]))[0]
            if deepest.size == 0:
                continue
            else:
                deepest = deepest[-1]
            deepest_pres = pres0[deepest]
            min_pres = pres0[j]-h/2
            max_pres = pres0[j]+h/2
            if min_pres<0:

                max_pres+=np.abs(min_pres)
                min_pres= 0
            if max_pres>deepest_pres:
                delta = max_pres-deepest_pres
                min_pres-=delta
                max_pres = np.copy(deepest_pres)

            jj = np.where( (pres0>=min_pres) & (pres0<=max_pres) & (np.isfinite(gamman[:,i])))[0]
            if jj.size == 0:
                continue
            try:
                pt = np.polyfit(gamman[jj,i],CT[jj,i],POLY_ORDER)
                ppt = np.poly1d(pt)
                CT_mean[j,i] = ppt(sg1s[j])
                ps = np.polyfit(gamman[jj,i],SA[jj,i],POLY_ORDER)
                pps = np.poly1d(ps)
                SA_mean[j,i] = pps(sg1s[j])
            except np.linalg.LinAlgError as error:
                polynomial_status[i] = 2
                stage_message[i] = "polynomial LinAlgError: " + str(error)
            
            
            
    MLD = mixed_layer_depth(Z, SG0)

    #NP = PRES.size
    dCTdz_mean = np.full( (NP, N_PROF), np.nan )
    dSAdz_mean = np.full( (NP, N_PROF), np.nan )

    N2_mean = np.full((NP, N_PROF), np.nan)
    for i in np.arange(N_PROF):
        dCTdz_mean[1:-1,i] = -(CT_mean[2:,i]-CT_mean[:-2,i])/(Zm[2:,i]-Zm[:-2,i])
        dSAdz_mean[1:-1,i] = -(SA_mean[2:,i]-SA_mean[:-2,i])/(Zm[2:,i]-Zm[:-2,i])

        #mCT = 0.5*(CT_mean[2:,i]+CT_mean[:-2,i])
        #mSA = 0.5*(SA_mean[2:,i]+SA_mean[:-2,i])
        #mP = 0.5*(PRES[2:]+PRES[:-2])
        #N2_mean[1:-1,i] = -9.81*(- dCT_mean[1:-1,i]*gsw.alpha(mSA,mCT,mP) + dSA_mean[1:-1,i]*gsw.beta(mSA,mCT,mP))


    chi = np.full(Krho_st.shape, np.nan)
    Pdia = np.full(Krho_st.shape, np.nan)

    chiS = np.full(Krho_st.shape, np.nan)
    PdiaS = np.full(Krho_st.shape, np.nan)

    chi_e = np.full(Krho_st.shape, np.nan)
    Pdia_e = np.full(Krho_st.shape, np.nan)

    chiS_e = np.full(Krho_st.shape, np.nan)
    PdiaS_e = np.full(Krho_st.shape, np.nan)

    gamman_fs = np.full(Krho_st.shape, np.nan)
    CT_fs = np.full(Krho_st.shape, np.nan)
    SA_fs = np.full(Krho_st.shape, np.nan)

    Ndataseg = np.full(Krho_st.shape, np.nan)
    mean_res = np.full(Krho_st.shape, np.nan)
    for i in range(N_PROF):
        for j in range(depth_bin.size):
            jj = np.abs(Zm[:,i]-depth_bin[j])<dZ/2
            Ndataseg[j,i] = np.sum(np.isfinite(CT[jj,i]))*window_size/dZ
            mean_res[j,i] = window_size/Ndataseg[j,i]

            CT_fs[j,i] = np.nanmean(CT[jj,i])
            SA_fs[j,i] = np.nanmean(SA[jj,i])
            gamman_fs[j,i] = np.nanmean(gamman[jj,i])
            #continues only when mean resolution is high (<10 m)
            if mean_res[j,i]>max([dz*2,min_resolution_micro]):
                eps_st[j,i] = np.nan
                Krho_st[j,i] = np.nan
            elif mean_res[j,i]<=max([dz*2,min_resolution_micro]):
                chi[j,i] = np.nanmean(2*Krho_st[j,i]*(dCTdz[jj,i])**2)
                Pdia[j,i] = np.nanmean(2*Krho_st[j,i]*(dCTdz_mean[jj,i])**2)

                chi_e[j,i] = np.nanmean(0.2*2*eps_st[j,i]*(dCTdz[jj,i])**2/N2[jj,i])
                Pdia_e[j,i] = np.nanmean(0.2*2*eps_st[j,i]*(dCTdz_mean[jj,i])**2/N2[jj,i])


                chiS[j,i] = np.nanmean(2*Krho_st[j,i]*(dSAdz[jj,i])**2)
                PdiaS[j,i] = np.nanmean(2*Krho_st[j,i]*(dSAdz_mean[jj,i])**2)

                chiS_e[j,i] = np.nanmean(0.2*2*eps_st[j,i]*(dSAdz[jj,i])**2/N2[jj,i])
                PdiaS_e[j,i] = np.nanmean(0.2*2*eps_st[j,i]*(dSAdz_mean[jj,i])**2/N2[jj,i])


    ds = xr.Dataset(
            data_vars = {
                "longitude" : ( ("time"),Lon, {"Description": "Longitude"} ),
                "latitude" : ( ("time"),Lat, {"Description": "Longitude"} ),
                "MLD" : ( ("time"),MLD, {"Description": "Mixed layer depth"} ),
                "pressure" : ( ("Z", "time"), P, {"Description": "Pressure", "Units": "dbar"} ),
                "Zmeas" : ( ("Z", "time"), Zm, {"Description": "Mean measurement depth", "Units": "m"} ),
                "CT" : ( ("Z", "time"), CT, {"Description": "Conservative Temperature", "Units": "degC"} ),
                "SA" : ( ("Z", "time"), SA, {"Description": "Salinity Absolute", "Units": "g/kg"} ),
                "CT_poly" : ( ("Z", "time"), CT_mean, {"Description": "Smooth Conservative Temperature", "Units": "degC"} ),
                "SA_poly" : ( ("Z", "time"), SA_mean, {"Description": "Smooth Salinity Absolute", "Units": "g/kg"} ),
                "gamman" : ( ("Z", "time"), gamman, {"Description": "Neutral density", "Units": "kg/m3"} ),
                "SG0" : ( ("Z", "time"), SG0, {"Description": "Potential Density", "Units": "kg/m3"} ),
                "spice0" : ( ("Z", "time"), Spi0, {"Description": "Spiciness", "Units": "kg/m3"} ),
                "T" : ( ("Z", "time"), TEMP, {"Description": "In situ Temperature", "Units": "degC"} ),
                "SP" : ( ("Z", "time"), PSAL, {"Description": "Practical Salinity", "Units": "PSU"} ),
                "dCTdz" : ( ("Z", "time"), dCTdz, {"Description": "Conservative Temperature vertical gradient", "Units": "degC/m"} ),
                "dSAdz" : ( ("Z", "time"), dSAdz, {"Description": "Absolute salinity vertical gradient", "Units": "(g/kg))/m"} ),
                "dCTdz_poly" : ( ("Z", "time"), dCTdz_mean, {"Description": "Smoothed Conservative Temperature vertical gradient", "Units": "degC/m"} ),
                "dSAdz_poly" : ( ("Z", "time"), dSAdz_mean, {"Description": "Smoothed Absolute salinity vertical gradient", "Units": "(g/kg))/m"} ),
                "N2" : (  ("Z", "time"), N2, {"Description": "Buoyancy frequency squared", "Units": "1/s^2"} ),
                "epsilon" : (  ("Zfs", "time"), eps_st, {"Description": "TKE dissipation", "Units": "W/kg"} ),
                "gamman_fs" : ( ("Zfs", "time"), gamman_fs, {"Description": "Neutral density in fine scale coordinates", "Units": "kg/m3"} ),
                "CT_fs" : ( ("Zfs", "time"), CT_fs, {"Description": "Conservative temperature in fine scale coordinates", "Units": "degC"} ),
                "SA_fs" : ( ("Zfs", "time"), CT_fs, {"Description": "Absolute salinity in fine scale coordinates", "Units": "g/kg"} ),
                "Krho" : (  ("Zfs", "time"), Krho_st, {"Description": "Diapycnal diffusivity", "Units": "m2/s"} ),
                "N_data_fs" : (  ("Zfs", "time"), Ndataseg, {"Description": "Number of datapoints per fine-scale segment", "Units": "#"} ),
                "mean_res" : (  ("Zfs", "time"), mean_res, {"Description": "Mean vertical resolution per fine-scale segment", "Units": "#"} ),
                "chi_fs" : (  ("Zfs", "time"), chi, {"Description": "Thermal variance dissipation from fine scale N2", "Units": "K^2/s"} ),
                "Pdia_fs" : (("Zfs", "time"), Pdia, {"Description": "Diapycnal thermal variance production from polynomial and fine scale N2 ", "Units": "K^2/s"} ),
                "chiS_fs" : ( ("Zfs", "time"), chiS, {"Description": "Haline variance dissipation from fine scale N2", "Units": "(g/kg)^2/s"} ),
                "PdiaS_fs" : ( ("Zfs", "time"), PdiaS,  {"Description": "Diapycnal haline variance production from polynomial and fine scale N2 ", "Units": "(g/kg)^2/s"} ),
                "chi" : (  ("Zfs", "time"), chi_e, {"Description": "Thermal variance dissipation", "Units": "K^2/s"} ),
                "Pdia" : (("Zfs", "time"), Pdia_e, {"Description": "Thermal variance production from polynomial (diapycnal)", "Units": "K^2/s"} ),
                "chiS" : ( ("Zfs", "time"), chiS_e, {"Description": "Haline variance dissipation", "Units": "(g/kg)^2/s"} ),
                "PdiaS" : ( ("Zfs", "time"), PdiaS_e, {"Description": "Haline variance production from polynomial (diapycnal)", "Units": "(g/kg)^2/s"} ),


            },
            coords = {
                "time":Time,
                "Z": Z,
                "Zfs": depth_bin,
            },
            attrs = {"Description":"Fine-scale mixing estimates from Argo floats; legacy-compatible per-profile stage"},
        )
    ds.Z.attrs['Description'] = 'Depth coordinate for fully resolved data (m)'
    ds.Zfs.attrs['Description'] = 'Depth coordinate for fine scale parameters (m)'
    ds.time.attrs['Description'] = 'Time coordinate'
    # Carry identities separately from time: global observations may share dates.
    platform = _profile_identity_values(DS, ("PLATFORM_NUMBER", "WMO"), iigood)
    cycle = _profile_identity_values(DS, ("CYCLE_NUMBER",), iigood)
    direction = _profile_identity_values(DS, ("DIRECTION",), iigood)
    source = _profile_identity_values(DS, ("SOURCE_FILE", "source_file"), iigood, DS.encoding.get("source", ""))
    profile_id = _profile_identity_values(DS, ("PROFILE_ID", "profile_id"), iigood)
    fallback_id = [f"{platform[i]}:{cycle[i]}:{direction[i]}:{str(Time[i])}" for i in range(N_PROF)]
    profile_id = np.array([profile_id[i] or fallback_id[i] for i in range(N_PROF)], dtype=str)
    for name, values in (("profile_id", profile_id), ("platform_number", platform),
                         ("cycle_number", cycle), ("direction", direction), ("source_file", source)):
        ds[name] = (("time",), values)
    ds["input_profile_index"] = (("time",), iigood.astype(np.int64))
    ds["mean_pressure_spacing_upper_1000"] = (("time",), mdpdz[iigood], {"units": "dbar"})
    ds["position_valid"] = (("time",), position_valid.astype(np.int8))
    for name, values, meanings in (
        ("gamma_n_status", gamma_status, "computed unsupported_latitude invalid_position no_samples failed_samples"),
        ("mixsea_status", mixsea_status, "computed insufficient_samples equator_unsupported invalid_input invalid_position no_finite_epsilon"),
        ("polynomial_status", polynomial_status, "computed no_gamma fit_failure"),
        ("n2_status", n2_status, "computed insufficient_gamma"),
    ):
        ds[name] = (("time",), values, {"flag_values": np.arange(len(meanings.split()), dtype=np.int16), "flag_meanings": meanings})
    ds["gamma_n_failed_sample_count"] = (("time",), gamma_failed_count)
    ds["stage_message"] = (("time",), np.asarray(stage_message, dtype=str))
    ds.attrs.update(
        input_profiles=int(N_PROF0), accepted_profiles=int(N_PROF),
        rejected_resolution_profiles=int(N_PROF0 - N_PROF),
        invalid_position_profiles=int(np.count_nonzero(~position_valid)),
        scientific_mode="legacy_compatibility",
        scientific_parameters=json.dumps({"dz": dz, "max_depth": max_depth,
            "min_mean_res_in_upper_1000m": min_mean_res_in_upper_1000m,
            "min_resolution_micro": min_resolution_micro, "window_size": window_size,
            "finescale_spacing": window_size // 4, "poly_window_dbar": h, "poly_order": POLY_ORDER}, sort_keys=True),
        legacy_conventions="CT_from_t receives practical salinity; SA_fs stores CT_fs; N2 uses sorted gamma_n; original gradient signs and MLD criterion retained",
        failure_handling="Unsupported gamma_n latitudes, nonpositive/error gamma_n results, invalid positions, equatorial Mixsea estimates and caught numerical-input failures are flagged; their unavailable metrics remain NaN",
    )
    return ds


"""Numerical plotting products formerly computed in notebook cells 20--24."""
import numpy as np
import xarray as xr


def add_plot_products(ds, bootstrap_samples=None):
    """Add plot-ready transformations and statistical profiles without I/O.

    ``ds`` is the regional summary on SG/TIME. ``bootstrap_samples`` maps chi,
    chiS, Pdia, PSdia, Piso, PSiso, and Krho to (SG, TIME, bootstrap) arrays.
    The result retains every existing variable unchanged. New names distinguish
    temporal spread, normal-approximation mean intervals, and pooled bootstrap
    percentiles, including the original 0.5th-percentile and NaN conventions.
    All reductions use the full saved TIME range, as in the source notebook.
    """
    result = ds.copy(deep=False)
    required = {'CT', 'SA', 'N2', 'chi', 'chiS', 'Pdia', 'PSdia', 'Piso',
                'PSiso', 'epsilon', 'Krho', 'KisoT', 'KisoS', 'dCTdz',
                'dSAdz', 'hgrCT', 'hgrSA', 'z'}
    missing = required - set(ds.data_vars)
    if missing:
        raise ValueError('Missing plotting-product inputs: ' + ', '.join(sorted(missing)))
    values = {name: np.asarray(ds[name].transpose('SG', 'TIME').values) for name in required}
    units = {name: ds[name].attrs.get('Units', ds[name].attrs.get('units', '')) for name in required}

    def add(name, array, dims, description, unit, formula, cells, convention=None):
        if name in result:
            raise ValueError(f'Plot product would overwrite existing variable: {name}')
        attributes = {'Description': description, 'Units': unit, 'formula': formula,
                      'source_notebook_cells': cells}
        if convention:
            attributes['legacy_convention'] = convention
        result[name] = xr.DataArray(array, dims=dims, attrs=attributes)

    def add_field(name, array, description, unit, formula, cells):
        values[name] = array
        units[name] = unit
        add(name, array, ('SG', 'TIME'), description, unit, formula, cells)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # Cell 20: anomalies, normalized diagnostics and production fractions.
        for name in ('CT', 'SA'):
            add_field(name + '_time_anomaly', values[name] - np.nanmean(values[name], axis=1, keepdims=True),
                      f'{name} anomaly from its mean over saved time windows', units[name],
                      f'{name} - nanmean({name}, TIME)', '20')
        for name in ('N2', 'hgrCT'):
            add_field(name + '_time_mean_ratio', values[name] / np.nanmean(values[name], axis=1, keepdims=True),
                      f'{name} divided by its mean over saved time windows', '1',
                      f'{name} / nanmean({name}, TIME)', '20' if name == 'N2' else '24')
        for name in ('chi', 'chiS', 'Piso', 'PSiso', 'Pdia', 'PSdia'):
            add_field(name + '_time_mean_log10_ratio',
                      np.log10(values[name] / np.nanmean(values[name], axis=1, keepdims=True)),
                      f'Base-10 logarithm of {name} divided by its time mean', '1',
                      f'log10({name} / nanmean({name}, TIME))', '20' if name in ('chi', 'chiS') else '24')
        for tracer, vertical, horizontal in [('CT', 'dCTdz', 'hgrCT'), ('SA', 'dSAdz', 'hgrSA')]:
            add_field(tracer + '_vertical_horizontal_gradient_ratio', values[vertical] / np.sqrt(values[horizontal]),
                      f'Vertical {tracer} gradient divided by horizontal gradient magnitude', '1',
                      f'{vertical} / sqrt({horizontal})', '20')
        for production, dissipation in [('Pdia', 'chi'), ('PSdia', 'chiS')]:
            add_field(production + '_dissipation_percent', values[production] / values[dissipation] * 100,
                      f'{production} as a percentage of {dissipation}', '%',
                      f'100 * {production} / {dissipation}', '20')
        for name in ('epsilon', 'Krho', 'KisoT'):
            add_field(name + '_log10', np.log10(values[name]),
                      f'Base-10 logarithm of {name} expressed in {units[name]}', '1',
                      f'log10(numerical {name} in {units[name]})', '20')

        # Cells 21, 22, 24 distinguish squared mean gradient from mean square.
        add_field('dCTdz_squared', values['dCTdz'] ** 2,
                  'Squared vertical conservative-temperature gradient', 'degC^2/m^2', 'dCTdz**2', '21,24')
        add_field('dCTdz_squared_mean_gradient_ratio',
                  values['dCTdz'] ** 2 / np.nanmean(values['dCTdz'], axis=1, keepdims=True) ** 2,
                  'Squared vertical CT gradient divided by the square of its temporal mean', '1',
                  'dCTdz**2 / nanmean(dCTdz, TIME)**2', '24')
        for name, unit in [('hgrCT', 'degC/m'), ('hgrSA', 'g/kg/m')]:
            add_field(name + '_magnitude', np.sqrt(values[name]),
                      f'Horizontal gradient magnitude from {name}', unit, f'sqrt({name})', '22,23')

        # Temporal summaries in cells 21--23. These average windows equally,
        # without area, observation-count, or overlapping-window corrections.
        mean_names = ('z', 'dCTdz', 'hgrCT', 'Pdia', 'Piso', 'chi', 'chiS',
                      'Krho', 'KisoT', 'KisoS', 'PSdia', 'PSiso', 'SA', 'CT',
                      'hgrSA_magnitude', 'hgrCT_magnitude')
        means = {}
        for name in mean_names:
            means[name] = np.nanmean(values[name], axis=1)
            add(name + '_time_mean', means[name], ('SG',),
                f'Mean {name} over saved time windows', units[name], f'nanmean({name}, TIME)', '21,22,23',
                'Equal weighting of overlapping time windows; nonfinite values handled by NumPy nanmean.')
        add('dCTdz_time_mean_squared', means['dCTdz'] ** 2, ('SG',),
            'Square of the time-mean vertical CT gradient', 'degC^2/m^2',
            'nanmean(dCTdz, TIME)**2', '21')
        for name in ('KisoT', 'KisoS'):
            add(name + '_time_median', np.nanmedian(values[name], axis=1), ('SG',),
                f'Median {name} over saved time windows', units[name], f'nanmedian({name}, TIME)', '21,22,23')
        for name in ('dCTdz_squared', 'hgrCT', 'Pdia', 'Piso', 'Krho', 'KisoT', 'chi', 'chiS'):
            for percentile, suffix in [(2.5, 'p02_5'), (97.5, 'p97_5')]:
                add(name + '_time_' + suffix, np.nanpercentile(values[name], percentile, axis=1), ('SG',),
                    f'{percentile:g}th percentile of {name} over time windows', units[name],
                    f'nanpercentile({name}, {percentile:g}, TIME)', '21',
                    'Temporal spread of overlapping-window estimates, not a bootstrap confidence interval.')
        for name in ('SA', 'CT', 'hgrSA_magnitude', 'hgrCT_magnitude', 'KisoT', 'KisoS'):
            for percentile, suffix in [(5, 'p05'), (95, 'p95')]:
                add(name + '_time_' + suffix, np.nanpercentile(values[name], percentile, axis=1), ('SG',),
                    f'{percentile:g}th percentile of {name} over time windows', units[name],
                    f'nanpercentile({name}, {percentile:g}, TIME)', '22,23',
                    'Temporal spread of overlapping-window estimates, not a bootstrap confidence interval.')
        for name in ('Krho', 'chi', 'Pdia', 'Piso', 'chiS', 'PSdia', 'PSiso'):
            count = np.sum(np.isfinite(values[name]), axis=1)
            standard_error = np.nanstd(values[name], axis=1) / np.sqrt(count)
            add(name + '_time_finite_count', count, ('SG',),
                f'Number of finite time-window {name} estimates', '1', f'sum(isfinite({name}), TIME)', '22')
            add(name + '_time_standard_error', standard_error, ('SG',),
                f'Legacy temporal standard error of mean {name}', units[name],
                f'nanstd({name}, TIME, ddof=0) / sqrt(sum(isfinite({name}), TIME))', '22',
                'Original formula treats time windows as independent; overlapping windows are not corrected.')
            for sign, suffix in [(-1, 'lo'), (1, 'hi')]:
                add(name + '_time_mean_normal95_' + suffix, means[name] + sign * 1.96 * standard_error, ('SG',),
                    f'Legacy mean {name} {"minus" if sign < 0 else "plus"} 1.96 temporal standard errors', units[name],
                    f'nanmean({name}, TIME) {"-" if sign < 0 else "+"} 1.96 * {name}_time_standard_error', '22',
                    'Legacy normal-approximation band; overlapping time windows are not corrected.')

        if bootstrap_samples is not None:
            expected = {'chi', 'chiS', 'Pdia', 'PSdia', 'Piso', 'PSiso', 'Krho'}
            missing_bootstrap = expected - set(bootstrap_samples)
            if missing_bootstrap:
                raise ValueError('Missing pooled bootstrap inputs: ' + ', '.join(sorted(missing_bootstrap)))
            for name in sorted(expected):
                boot = np.asarray(bootstrap_samples[name])
                if boot.ndim != 3 or boot.shape[:2] != (ds.sizes['SG'], ds.sizes['TIME']):
                    raise ValueError(f'{name} bootstrap requires dimensions (SG, TIME, bootstrap)')
                add(name + '_pooled_bootstrap_mean', np.nanmean(boot, axis=(1, 2)), ('SG',),
                    f'Mean {name} over pooled time windows and bootstrap draws', units[name],
                    f'nanmean({name}_bootstrap, (TIME, bootstrap))', '15,23',
                    'Raw bootstrap residuals retain negative values; the time-window Piso/PSiso means use legacy positive floors.')
                lower = 2.5 if name in ('Piso', 'PSiso') else .5
                for percentile, suffix in [(lower, 'p02_5' if lower == 2.5 else 'p00_5'), (97.5, 'p97_5')]:
                    add(name + '_pooled_bootstrap_' + suffix, np.percentile(boot, percentile, axis=(1, 2)), ('SG',),
                        f'Legacy {percentile:g}th percentile of pooled {name} bootstrap draws', units[name],
                        f'percentile({name}_bootstrap, {percentile:g}, (TIME, bootstrap))', '15' if name in ('Piso', 'PSiso') else '23',
                        'Ordinary percentile propagates any NaN in pooled samples. Cell 23 uses 0.5, not 2.5, for this lower bound.' if lower == .5 else
                        'Ordinary percentile propagates any NaN in pooled samples; original cell 15 uses 2.5 and 97.5.')
    result.attrs = dict(result.attrs)
    result.attrs['plot_products_source'] = 'plots_mixing_NATRE_gamman_analysis.ipynb cells 20-24 (pooled residual bounds also cell 15)'
    result.attrs['plot_products_conventions'] = (
        'All temporal summaries cover the saved TIME range, equally weight overlapping windows, and preserve legacy NumPy formulas. '
        'Temporal percentile spreads, mean +/- 1.96*SE bands and pooled-bootstrap percentiles are separate products. '
        'Cell 23 lower pooled percentiles are 0.5 except residuals at 2.5; ordinary pooled percentile propagates NaNs. '
        'Cell 23 Piso/PSiso plotted means are inherited from cell 22 (floored time means), while their bounds use unfloored pooled residual draws. '
        'Zero denominators and nonpositive logarithms retain NumPy inf/NaN behavior.'
    )
    result.attrs['pooled_bootstrap_plot_products_saved'] = int(bootstrap_samples is not None)
    return result


"""Faithful regional analysis port of plots_mixing_NATRE_gamman_analysis.ipynb.

All numerical legacy conventions are retained by default. The only unconditional
repair is the invalid-data return shape of bootstrap_Krho (four NaNs rather than
an unpacking exception when bootstrap samples are requested). Global geometry
and Coriolis guards are explicit opt-ins. This module performs no I/O on import.
"""
import json
import numpy as np
import xarray as xr
import scipy.interpolate as intrp
from scipy.stats import linregress
from scipy.stats import t as tdist
import gsw
import seawater as sw

def add_quicklook_products(summary, time, lon, lat, z, finescale_z, mean_res,
                           chi_missing_before_clip, epsilon_cleaned, outliers,
                           epsilon_depth, krho_depth, gamma_mix):
    """Save numerical QC/quicklook products from notebook cells 6, 9--11, 17."""
    result = summary.copy(deep=False)
    edges = np.array([0, 2.5, 7.5, 15, 30, 1e23])
    counts = np.zeros(5, dtype=np.int64)
    missing_counts = np.zeros(5, dtype=np.int64)
    percentages = np.zeros(5, dtype=np.int64)
    for index in range(5):
        selected = (mean_res >= edges[index]) & (mean_res < edges[index + 1])
        counts[index] = np.count_nonzero(selected)
        missing_counts[index] = np.count_nonzero(selected & chi_missing_before_clip)
        if counts[index]:
            percentages[index] = missing_counts[index] / counts[index] * 100
    result = result.assign_coords(resolution_bin=np.arange(5), Z=z,
                                  Zfs=finescale_z, profile=np.arange(len(time)))
    result['resolution_bin_lower'] = ('resolution_bin', edges[:-1])
    result['resolution_bin_upper'] = ('resolution_bin', edges[1:])
    result['resolution_sample_count'] = ('resolution_bin', counts)
    result['resolution_missing_chi_count'] = ('resolution_bin', missing_counts)
    result['resolution_missing_chi_percent'] = ('resolution_bin', percentages,
        {'Units': '%', 'legacy_convention': 'Cell 6 truncates percentages to integer values; zero for empty bins.'})
    finite = mean_res[np.isfinite(mean_res)]
    result['median_finescale_resolution'] = xr.DataArray(np.median(finite) if finite.size else np.nan,
                                                        attrs={'Units': 'm'})
    result['profile_time'] = ('profile', time)
    result['profile_longitude'] = ('profile', lon)
    result['profile_latitude'] = ('profile', lat)
    result['finescale_mean_resolution'] = (('Zfs', 'profile'), mean_res, {'Units': 'm'})
    result['finescale_outlier_mask'] = (('Zfs', 'profile'), outliers.astype('int8'))
    result['epsilon_finescale_cleaned'] = (('Zfs', 'profile'), epsilon_cleaned, {'Units': 'W/kg'})
    with np.errstate(divide='ignore', invalid='ignore'):
        result['epsilon_finescale_cleaned_log10'] = (('Zfs', 'profile'), np.log10(epsilon_cleaned))
        result['epsilon_depth_mean'] = ('Z', np.nanmean(epsilon_depth, axis=1), {'Units': 'W/kg'})
        result['epsilon_Krho_stratification_proxy_depth_mean'] = (
            'Z', np.nanmean(epsilon_depth / krho_depth * gamma_mix, axis=1), {'Units': '1/s^2'})
    contour_sa = np.arange(35, 38, .05)
    contour_ct = np.arange(3, 29, .5)
    mesh_sa, mesh_ct = np.meshgrid(contour_sa, contour_ct)
    result = result.assign_coords(SA_contour=contour_sa, CT_contour=contour_ct)
    result['sigma0_TS_contour'] = (('CT_contour', 'SA_contour'), gsw.sigma0(mesh_sa, mesh_ct),
        {'Units': 'kg/m^3', 'Description': 'Legacy NATRE T-S plot background; not a geographic or data filter'})
    for component in ['x', 'y']:
        result['iso_slope_' + component + '_nonsignificant'] = (
            result['iso_slope_' + component + '_pvalue'] > .05).astype('int8')
    result['iso_slope_x_manual_nonsignificant'] = (result.iso_slope_x_manual_pvalue > .05).astype('int8')
    result.attrs['quicklook_products_source'] = 'Original notebook cells 6, 9-11, 17; original formulas and histogram integer truncation retained'
    return result


def estimate_vertical_corr_length(eps_profile, z, min_valid=10):
    """Notebook helper, retained for exploratory analysis (not used by defaults)."""
    eps_profile = np.array(eps_profile)
    z = np.array(z)
    mask = np.isfinite(eps_profile)
    if np.sum(mask) < min_valid:
        return np.nan
    eps_valid = eps_profile[mask]
    z_valid = z[mask]
    dz = np.nanmedian(np.diff(z_valid))
    eps_anom = eps_valid - np.nanmean(eps_valid)
    ac = np.correlate(eps_anom, eps_anom, mode='full')
    ac = ac[ac.size // 2:] / ac[ac.size // 2]
    for lag_idx, val in enumerate(ac):
        if val < 1 / np.e:
            return lag_idx * dz
    return np.nan

# Bounded bootstrap batches preserve the legacy random draws and arithmetic.
BOOTSTRAP_BATCH_ROWS = 32

BOOTSTRAP_MAX_SAMPLE_BYTES = 8 * 1024 * 1024

def _bootstrap_sample_means(values, n_boot, n_eff, rng, *, nanmean=False, indices=False):
    """Draw a bounded number of complete, contiguous bootstrap rows at a time."""
    rng = np.random if rng is None else rng
    reduction = np.nanmean if nanmean else np.mean
    mean_dtype = np.asarray(reduction(values[:1])).dtype
    means = np.empty(n_boot, dtype=mean_dtype)
    # Account for both index and value matrices. A single replicate is the
    # smallest batch, exactly as in the legacy loop.
    bytes_per_row = max(1, n_eff) * (np.dtype(np.intp).itemsize + values.dtype.itemsize)
    rows = max(1, min(BOOTSTRAP_BATCH_ROWS, BOOTSTRAP_MAX_SAMPLE_BYTES // bytes_per_row))
    for start in range(0, n_boot, rows):
        count = min(rows, n_boot - start)
        if indices:
            chosen = rng.choice(len(values), size=(count, n_eff), replace=True)
            samples = values[chosen]
        else:
            samples = rng.choice(values, size=(count, n_eff), replace=True)
        means[start:start + count] = reduction(samples, axis=1)
        # Release this batch before allocating the next one, so retained
        # previous buffers do not increase the temporary-memory peak.
        del samples
        if indices:
            del chosen
    return means

def _legacy_scale_means(means, multiplier, denominator):
    """Retain the two scalar arithmetic promotions and their rounding points."""
    scalar_zero = np.zeros((), dtype=means.dtype)[()]
    product_zero = multiplier * scalar_zero
    product_dtype = np.asarray(product_zero).dtype
    quotient_dtype = np.asarray(product_zero / denominator).dtype
    products = np.multiply(multiplier, means, dtype=product_dtype)
    return np.divide(products, denominator, dtype=quotient_dtype)

def bootstrap_CI_var(data, n_boot=100, n_eff=None, return_bootstrap=False, rng=None):
    data = np.array(data)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        if return_bootstrap:
            return np.nan, np.nan, np.nan, np.full(n_boot, np.nan)
        return np.nan, np.nan, np.nan
    if n_eff is None or n_eff > len(data):
        n_eff = len(data)
    boot_vals = _bootstrap_sample_means(data, n_boot, n_eff, rng)
    result = (np.mean(boot_vals), np.percentile(boot_vals, 2.5), np.percentile(boot_vals, 97.5))
    return result + (boot_vals,) if return_bootstrap else result

def bootstrap_Krho(eps_vals, N2_val, gammamix=.2, n_boot=100, n_eff=None, return_bootstrap=False, rng=None):
    eps = np.array(eps_vals)
    mask = np.isfinite(eps)
    if np.sum(mask) < 10 or not np.isfinite(N2_val) or N2_val <= 0:
        if return_bootstrap:
            return np.nan, np.nan, np.nan, np.full(n_boot, np.nan)
        return np.nan, np.nan, np.nan
    eps = eps[mask]
    if n_eff is None or n_eff > len(eps):
        n_eff = len(eps)
    means = _bootstrap_sample_means(eps, n_boot, n_eff, rng, indices=True)
    finite = np.isfinite(means)
    if not np.any(finite):
        # The original list contains only Python np.nan in this branch.
        boot_vals = np.full(n_boot, np.nan)
    else:
        boot_vals = _legacy_scale_means(means, gammamix, N2_val)
        if not np.all(finite):
            # A mixed list of NumPy scalars and Python np.nan uses this dtype.
            boot_vals = boot_vals.astype(np.result_type(boot_vals.dtype, float), copy=False)
            boot_vals[~finite] = np.nan
    result = (np.nanmean(boot_vals), np.nanpercentile(boot_vals, 2.5), np.nanpercentile(boot_vals, 97.5))
    return result + (boot_vals,) if return_bootstrap else result

def bootstrap_Pdia(eps_vals, grad_smooth, N2_avg, gammamix=.2, n_boot=100, n_eff=None, return_bootstrap=False, rng=None):
    eps_vals = np.array(eps_vals)
    eps_vals = eps_vals[np.isfinite(eps_vals)]
    if len(eps_vals) < 5 or not np.isfinite(N2_avg) or N2_avg <= 0 or not np.isfinite(grad_smooth):
        if return_bootstrap:
            return np.nan, np.nan, np.nan, np.full(n_boot, np.nan)
        return np.nan, np.nan, np.nan
    if n_eff is None or n_eff > len(eps_vals):
        n_eff = len(eps_vals)
    means = _bootstrap_sample_means(eps_vals, n_boot, n_eff, rng, nanmean=True)
    multiplier = 2 * grad_smooth**2 * gammamix
    boot_vals = _legacy_scale_means(means, multiplier, N2_avg)
    result = (np.nanmean(boot_vals), np.nanpercentile(boot_vals, 2.5), np.nanpercentile(boot_vals, 97.5))
    return result + (boot_vals,) if return_bootstrap else result

#interpolates mixing parameters to finer grid
def interpolate_high(Z,X,z, log_trans = True):
    x = np.full((z.size, X.shape[1]), np.nan)
    if log_trans:
        X = np.log10(X)
    for i in range(X.shape[1]):
        iif = np.isfinite(X[:,i])
        if np.sum(iif)>0:
            XX = X[iif,i]
            ZZ = Z[iif]
            intx = intrp.interp1d(ZZ,XX, bounds_error = False)
            x[:,i] = intx(z)
    if log_trans:
        x = 10**x
    return x
    


def aggregate_region(ds: xr.Dataset, config=None) -> xr.Dataset:
    """Apply the notebook's complete analysis to one bounded spatial region.

    Expected input dimensions are (Z, time) and (Zfs, time), as produced by the
    profile stage. Call this on a bounded region, never on the entire archive.
    The input is not mutated. The result has (SG, TIME) dimensions, with optional
    bootstrap samples (SG, TIME, bootstrap).

    Supported config keys (legacy defaults in parentheses):
      seed (None); n_boot (1000); seasonal_mode (False);
      time_origin ('2014-01-01T00:00:00.0000'); nmonths (3);
      nmonths_window (12.0); time_centers_days (None);
      start_time/end_time (None, inclusive profile filters);
      gamma_mix (0.2); density_min/density_max (26.75/28.05);
      density_step (0.05); density_centers (None; overrides range);
      correlation_length_m (4.0); vertical_spacing_m (2.0);
      reference_density (28.0); include_bootstrap (False);
      global_mode (False); min_abs_coriolis (1e-6, global_mode only).

    Changing gamma_mix retains the original default semantics: notebook helper
    defaults used 0.2, so all bootstrap calls now explicitly share this setting.
    Legacy overlapping density windows and seasonal time selection are retained.
    global_mode unwraps local longitudes across the dateline and masks velocity
    estimates when |f| is below min_abs_coriolis or the reference is unavailable.
    Configuration is recorded in output attributes for reproducibility.
    """
    config = dict(config or {})
    if 'start_time' in config and config['start_time'] is not None:
        ds = ds.isel(time=np.flatnonzero(ds.time.values >= np.datetime64(config['start_time'])))
    if 'end_time' in config and config['end_time'] is not None:
        ds = ds.isel(time=np.flatnonzero(ds.time.values <= np.datetime64(config['end_time'])))
    if not ds.sizes.get('time', 0):
        raise ValueError('aggregate_region requires at least one profile in the selected period')
    required = ['time', 'Z', 'Zfs', 'pressure', 'latitude', 'longitude', 'chi',
                'Pdia', 'chiS', 'PdiaS', 'CT', 'SA', 'dCTdz', 'dSAdz', 'CT_poly',
                'SA_poly', 'Krho', 'epsilon', 'gamman', 'mean_res']
    missing = [name for name in required if name not in ds]
    if missing:
        raise ValueError('Missing regional-analysis input variables: ' + ', '.join(missing))
    time = ds.time.values.astype('datetime64[ns]')
    z = ds.Z.values
    Z = ds.Zfs.values
    def profile_values(name, depth_dim=None):
        arr = ds[name]
        if depth_dim:
            arr = arr.transpose(depth_dim, 'time')
        else:
            arr = arr.transpose('time')
        return np.array(arr.values, copy=True)
    pres = profile_values('pressure', 'Z')
    lat = profile_values('latitude')
    lon = profile_values('longitude')
    lon_original = lon.copy()
    global_mode = bool(config.get('global_mode', False))
    if global_mode:
        center = np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(lon))),
                                       np.nanmean(np.cos(np.radians(lon)))))
        lon = center + (lon - center + 180) % 360 - 180
    chi0 = profile_values('chi', 'Zfs')
    Pdia0 = profile_values('Pdia', 'Zfs')
    chiS0 = profile_values('chiS', 'Zfs')
    PdiaS0 = profile_values('PdiaS', 'Zfs')
    Krho0 = profile_values('Krho', 'Zfs')
    epsilon0 = profile_values('epsilon', 'Zfs')
    mean_res = profile_values('mean_res', 'Zfs')
    CT = profile_values('CT', 'Z')
    SA = profile_values('SA', 'Z')
    dCTdz = profile_values('dCTdz', 'Z')
    dSAdz = profile_values('dSAdz', 'Z')
    CT_poly = profile_values('CT_poly', 'Z')
    SA_poly = profile_values('SA_poly', 'Z')
    gamman = profile_values('gamman', 'Z')
    lat_range = [lat.min(), lat.max()]
    lon_range = [lon.min(), lon.max()]
    GAMMA_mix = float(config.get('gamma_mix', 0.2))
    SEASONAL_MODE = bool(config.get('seasonal_mode', False))
    nmonths = float(config.get('nmonths', 3))
    nmonths_window = float(config.get('nmonths_window', 12.0))
    time_origin = str(config.get('time_origin', '2014-01-01T00:00:00.0000'))
    L_eps = float(config.get('correlation_length_m', 4.0))
    dz = float(config.get('vertical_spacing_m', 2.0))
    dSG = float(config.get('density_step', 0.05))
    if min(nmonths, nmonths_window, L_eps, dz, dSG) <= 0:
        raise ValueError('Analysis window, grid and correlation scales must be positive')
    n_boot_i = int(config.get('n_boot', 1000))
    if n_boot_i < 1:
        raise ValueError('n_boot must be positive')
    seed = config.get('seed')
    rng = np.random if seed is None else np.random.RandomState(int(seed))
    chi_missing_before_clip = np.isnan(chi0).copy()
    #cleans extreme data
    epsilonSD = np.nanstd(np.log10(epsilon0), axis = 1, keepdims = True)
    epsilonM = np.nanmean(np.log10(epsilon0), axis = 1, keepdims = True)

    baddata = epsilon0>10**(epsilonM+3*epsilonSD)
    #baddata  = (baddata) | (mean_res>2)
    chi0[baddata] = np.nan
    Pdia0[baddata] = np.nan


    chiS0[baddata] = np.nan
    PdiaS0[baddata] = np.nan

    epsilon0[baddata] = np.nan
    Krho0[baddata] = np.nan
    chi= interpolate_high(Z,chi0,z)
    Pdia = interpolate_high(Z,Pdia0,z)
    chiS = interpolate_high(Z,chiS0,z)
    PdiaS = interpolate_high(Z,PdiaS0,z)
    epsilon = interpolate_high(Z,epsilon0,z)
    Krho = interpolate_high(Z,Krho0,z)
    #defines parameters for averaging

    if not SEASONAL_MODE:
        timer = (time - np.datetime64(time_origin)).astype(float)/60/60/24/1e9
        time_max = np.round(np.max(timer)/365.25)*365.25
        TIME = np.arange(365.25/12.*nmonths/2, time_max+365.25/12.*nmonths/2,365.25/12.*nmonths)
    else:
        timer = (time.astype('datetime64[M]').astype(int) % 12 + 1)*365.25/12
        TIME = np.arange(0.5,12.5,1)*365.25/12.
        nmonths_window = 1.


    if config.get('time_centers_days') is not None:
        TIME = np.asarray(config['time_centers_days'], dtype=float)
    if TIME.ndim != 1 or not np.all(np.isfinite(TIME)):
        raise ValueError('time_centers_days must be a finite 1D sequence')
    if not TIME.size:
        raise ValueError('The legacy calendar produced no time windows; configure time_centers_days or an earlier time_origin')
    #Define isopycnal bin edges and centers 
    SGi = np.arange(float(config.get('density_min', 26.75))-dSG/2,
                    float(config.get('density_max', 28.05))+dSG/2, dSG)
    SG = 0.5 * (SGi[1:] + SGi[:-1])
    if config.get('density_centers') is not None:
        SG = np.asarray(config['density_centers'], dtype=float)
    if SG.ndim != 1 or not SG.size or not np.all(np.isfinite(SG)) or np.any(np.diff(SG) <= 0):
        raise ValueError('density centers must be finite, increasing and nonempty')

    #Initialize output arrays with shape (density bins, time steps)
    shape = (SG.size, TIME.size)
    Nprof_sg = np.full(TIME.size, np.nan)   #number of profiles per time bin
    #Nprof_bin_sg = np.full(shape, np.nan)   #number of profiles per density bin and time
    n = np.full(shape, np.nan)
    Nprof_bin_sg = np.zeros(shape, dtype=np.int64)
    Nobs_bin_sg = np.zeros(shape, dtype=np.int64)
    N_eff_sg = np.zeros(shape, dtype=np.int64)

    # Depth and pressure
    z_sg = np.full(shape, np.nan)
    p_sg = np.full(shape, np.nan)
    h_sg = np.full(shape, np.nan) 

    # Hydrographic properties
    CT_sg = np.full(shape, np.nan)
    SA_sg = np.full(shape, np.nan)
    var_CT_sg = np.full(shape, np.nan)
    var_SA_sg = np.full(shape, np.nan)

    #Position
    lat_sg = np.full(TIME.size, np.nan)
    lon_sg = np.full(TIME.size, np.nan)

    #Property gradients
    dCTdz_sg = np.full(shape, np.nan)
    dSAdz_sg = np.full(shape, np.nan)

    dCTdx_sg = np.full(shape, np.nan)
    dCTdy_sg = np.full(shape, np.nan)
    dSAdx_sg = np.full(shape, np.nan)
    dSAdy_sg = np.full(shape, np.nan)

    dCTdx_sg_err = np.full(shape, np.nan)
    dCTdy_sg_err = np.full(shape, np.nan)
    dSAdx_sg_err = np.full(shape, np.nan)
    dSAdy_sg_err = np.full(shape, np.nan)

    dCTdx_sg_p = np.full(shape, np.nan)
    dCTdy_sg_p = np.full(shape, np.nan)
    dSAdx_sg_p = np.full(shape, np.nan)
    dSAdy_sg_p = np.full(shape, np.nan)

    #Isopycnal slopes
    sx_sg = np.full(shape, np.nan)
    sy_sg = np.full(shape, np.nan)

    sx_sg_err = np.full(shape, np.nan)
    sy_sg_err = np.full(shape, np.nan)

    sx_sg_p = np.full(shape, np.nan)
    sy_sg_p = np.full(shape, np.nan)

    sx_t_stat = np.full(shape, np.nan)
    sx_pval = np.full(shape, np.nan)

    #Mixing diagnostics
    epsilon_sg   = np.full(shape, np.nan)
    Krho_sg = np.full(shape, np.nan)

    chi_sg   = np.full(shape, np.nan)
    chiS_sg   = np.full(shape, np.nan)

    Pdia_sg = np.full(shape, np.nan)
    PdiaS_sg = np.full(shape, np.nan)

    Pdia_poly_sg   = np.full(shape, np.nan)
    PdiaS_poly_sg   = np.full(shape, np.nan)

    Piso_sg = np.full(shape, np.nan)
    PSiso_sg = np.full(shape, np.nan)

    Piso_poly_sg = np.full(shape, np.nan)
    PSiso_poly_sg = np.full(shape, np.nan)

    N2_sg      = np.full(shape, np.nan)
    Lmixing_sg = np.full(shape, np.nan)

    #############################################################
    #uncertainties
    chi_sg_lo = np.full(shape, np.nan)
    chi_sg_hi = np.full(shape, np.nan)

    chiS_sg_lo = np.full(shape, np.nan)
    chiS_sg_hi = np.full(shape, np.nan)

    epsilon_sg_lo = np.full(shape, np.nan)
    epsilon_sg_hi = np.full(shape, np.nan)

    Krho_sg_lo = np.full(shape, np.nan)
    Krho_sg_hi = np.full(shape, np.nan)


    Pdia_poly_sg_lo = np.full(shape, np.nan)
    Pdia_poly_sg_hi = np.full(shape, np.nan)

    Piso_poly_sg_lo = np.full(shape, np.nan)
    Piso_poly_sg_hi = np.full(shape, np.nan)

    PdiaS_poly_sg_lo = np.full(shape, np.nan)
    PdiaS_poly_sg_hi = np.full(shape, np.nan)

    PSiso_poly_sg_lo = np.full(shape, np.nan)
    PSiso_poly_sg_hi = np.full(shape, np.nan)

    Pdia_sg_lo = np.full(shape, np.nan)
    Pdia_sg_hi = np.full(shape, np.nan)

    PdiaS_sg_lo = np.full(shape, np.nan)
    PdiaS_sg_hi = np.full(shape, np.nan)

    Piso_sg_lo = np.full(shape, np.nan)
    Piso_sg_hi = np.full(shape, np.nan)

    PSiso_sg_lo = np.full(shape, np.nan)
    PSiso_sg_hi = np.full(shape, np.nan)

    #############################################################
    #bootstrap samples
    chi_sg_boot_all = np.full((SG.size,TIME.size, n_boot_i),np.nan)
    chiS_sg_boot_all = np.full((SG.size,TIME.size, n_boot_i),np.nan)
    Pdia_sg_boot_all = np.full((SG.size,TIME.size, n_boot_i),np.nan)
    PdiaS_sg_boot_all = np.full((SG.size,TIME.size, n_boot_i),np.nan)
    Piso_sg_boot_all = np.full((SG.size,TIME.size, n_boot_i),np.nan)
    PSiso_sg_boot_all = np.full((SG.size,TIME.size, n_boot_i),np.nan)
    Krho_sg_boot_all = np.full((SG.size,TIME.size, n_boot_i),np.nan)

    for j in range(TIME.size):
    
        #time window for averaging
        time_window = 365.25 / 12 * nmonths_window / 2
        iip = np.abs(timer - TIME[j]) <= time_window
        nd  = np.sum(iip).astype(float) #number of profiles
        Nprof_sg[j] = nd

        #mean position of profiles in time window
        lat_sg[j] = np.nanmean(lat[iip])
        lon_sg[j] = np.nanmean(lon[iip])
    
        # Retain original profile order. A local depth-major boolean selection
        # yields exactly the same sample sequence as the legacy full-array mask.
        if not np.any(iip):
            continue
        window_gamman = gamman[:, iip]
        window_chi = chi[:, iip]
        window_chiS = chiS[:, iip]
        window_pres = pres[:, iip]
        window_CT = CT[:, iip]
        window_SA = SA[:, iip]
        window_CT_poly = CT_poly[:, iip]
        window_SA_poly = SA_poly[:, iip]
        window_epsilon = epsilon[:, iip]
        window_dCTdz = dCTdz[:, iip]
        window_dSAdz = dSAdz[:, iip]
        window_Pdia = Pdia[:, iip]
        window_PdiaS = PdiaS[:, iip]
        window_zz = np.broadcast_to(z[:, None], window_gamman.shape)
        window_lat2d = np.broadcast_to(lat[iip], window_gamman.shape)
        window_lon2d = np.broadcast_to(lon[iip], window_gamman.shape)

        for i in range(SG.size):
            #select data points in isopycnal bin, time window, and with valid mixing data
            ii = ((np.abs(window_gamman - SG[i]) <= dSG) & np.isfinite(window_chi + window_chiS))
            Nprof_bin_sg[i, j] = np.count_nonzero(np.any(ii, axis=0))
            Nobs_bin_sg[i, j] = np.count_nonzero(ii)
            nd_bin = np.sum(ii).astype(float)  #data points in the isopycnal layer & time window
    
            if np.sum(ii) == 0:
                continue

            if nd_bin > 0:
                n_eff = int(np.round(nd_bin / (2 * L_eps / dz)))
            else:
                n_eff =  max(1, int(nd_bin)) #1

            N_eff_sg[i, j] = n_eff
            #average hydrography and mixing variables
            z_sg[i, j] = np.nanmean(window_zz[ii])
            p_sg[i, j] = np.nanmean(window_pres[ii])
            CT_sg[i, j] = np.mean(window_CT[ii])
            SA_sg[i, j] = np.mean(window_SA[ii])

            #variances
            var_CT_sg[i, j] = np.nanvar(window_CT[ii] - window_CT_poly[ii])
            var_SA_sg[i, j] = np.nanvar(window_SA[ii] - window_SA_poly[ii])

            chi_sg[i, j], chi_sg_lo[i, j], chi_sg_hi[i, j], chi_sg_boot_all[i,j,:] = bootstrap_CI_var(window_chi[ii], n_boot=n_boot_i, n_eff=n_eff, return_bootstrap=True, rng=rng)    
            chiS_sg[i, j], chiS_sg_lo[i, j], chiS_sg_hi[i, j], chiS_sg_boot_all[i,j,:] = bootstrap_CI_var(window_chiS[ii], n_boot=n_boot_i, n_eff=n_eff, return_bootstrap=True, rng=rng)
     
            epsilon_sg[i, j], epsilon_sg_lo[i, j], epsilon_sg_hi[i, j] = bootstrap_CI_var(window_epsilon[ii], n_boot=n_boot_i, n_eff=n_eff, rng=rng)

            #isopycnal layers
            iitop    = (np.abs(window_gamman - (SG[i] - dSG/2)) < dSG/2)
            iibottom = (np.abs(window_gamman - (SG[i] + dSG/2)) < dSG/2)

            h_sg[i,j] = -(np.nanmean(window_zz[iitop])-np.nanmean(window_zz[iibottom])) #layer thickness

            #uses direct average of gradient instead of gradient of average
            dCTdz_sg[i,j] = -np.nanmean(window_dCTdz[ii])
            dSAdz_sg[i,j] = -np.nanmean(window_dSAdz[ii])

            grav = gsw.grav(lat_sg[j], p_sg[i,j])  # local gravity
            alpha = gsw.alpha(SA_sg[i,j], CT_sg[i,j], p_sg[i,j])
            beta  = gsw.beta(SA_sg[i,j], CT_sg[i,j], p_sg[i,j])
            N2_sg[i,j] = - grav * (-dCTdz_sg[i,j]*alpha + dSAdz_sg[i,j]*beta)

            #number of profiles normalized by layer thickness
            #Nprof_bin_sg[i, j] = 2 * np.count_nonzero(ii) / h_sg[i, j]

            #diapycnal production for window_CT and window_SA
            #Pdia_poly_sg[i, j]   = np.mean(window_Pdia[ii])
        
            #PdiaS_poly_sg[i, j]   = np.mean(window_PdiaS[ii])
            #s_PdiaS_poly_sg[i, j] = np.std(window_PdiaS[ii]) / np.sqrt(nd)
        
            #estimate horizontal gradients and isopycnal slopes if spatial coverage is good
            lat_range0 = np.nanmax(window_lat2d[ii]) - np.nanmin(window_lat2d[ii])
            lon_range0 = np.nanmax(window_lon2d[ii]) - np.nanmin(window_lon2d[ii])

            if ((np.sum(ii) > 10) &
                (nd > 10) &
                (lat_range0 > (lat_range[1] - lat_range[0]) / 3.) &
                (lon_range0 > (lon_range[1] - lon_range[0]) / 3.)):
            
                x = window_lon2d[ii]
                y = window_lat2d[ii]
                coslat = np.cos(np.nanmean(y) * np.pi / 180)
                factorx = 111e3*coslat
                factory = 111e3

                n[i,j] = np.sum(ii)
            
                #horizontal gradient of conservative temperature
                resCTx = linregress(x, window_CT[ii])
                resCTy = linregress(y, window_CT[ii])
                dCTdx_sg[i, j] = resCTx.slope / factorx
                dCTdy_sg[i, j] = resCTy.slope / factory
                dCTdx_sg_err[i, j] = resCTx.stderr / factorx
                dCTdy_sg_err[i, j] = resCTy.stderr / factory
                dCTdx_sg_p[i, j] = resCTx.pvalue
                dCTdy_sg_p[i, j] = resCTy.pvalue

                #horizontal gradient of absolute salinity
                resSAx = linregress(x, window_SA[ii])
                resSAy = linregress(y, window_SA[ii])
                dSAdx_sg[i, j] = resSAx.slope / factorx
                dSAdy_sg[i, j] = resSAy.slope / factory
                dSAdx_sg_err[i, j] = resSAx.stderr / factorx
                dSAdy_sg_err[i, j] = resSAy.stderr / factory
                dSAdx_sg_p[i, j] = resSAx.pvalue
                dSAdy_sg_p[i, j] = resSAy.pvalue
            
                #isopycnal slopes
                ressx = linregress(x, -window_zz[ii])
                ressy = linregress(y, -window_zz[ii])
                sx_sg[i, j] = ressx.slope / factorx
                sy_sg[i, j] = ressy.slope / factory
                sx_sg_err[i, j] = ressx.stderr / factorx
                sy_sg_err[i, j] = ressy.stderr / factory
                sx_sg_p[i, j] = ressx.pvalue
                sy_sg_p[i, j] = ressy.pvalue

                sx_t_stat[i,j] = sx_sg[i,j] / sx_sg_err[i,j] #manual check pvalue
                dof = n[i,j] - 2 #n - parameters (in this case, parameters are two: intercept + slope, as it is a regression)
                pval = 2*tdist.sf(abs(sx_t_stat[i,j]),dof)
                sx_pval[i,j] = pval

                '''
                #horizontal gradient of conservative temperature
                pxCT, VxCT = np.polyfit(x, window_CT[ii], 1, cov=True)
                pyCT, VyCT = np.polyfit(y, window_CT[ii], 1, cov=True)
                dCTdx_sg[i, j] = pxCT[0] / factorx
                dCTdy_sg[i, j] = pyCT[0] / factory
                dCTdx_sg_err[i, j] = np.sqrt(VxCT[0, 0]) / factorx
                dCTdy_sg_err[i, j] = np.sqrt(VyCT[0, 0]) / factory

                #horizontal gradient of absolute salinity
                dSAdx_sg[i, j] = np.polyfit(x, window_SA[ii], 1)[0] / (111e3 * coslat)
                dSAdy_sg[i, j] = np.polyfit(y, window_SA[ii], 1)[0] / 111e3

                #isopycnal slopes
                sx_sg[i, j] = np.polyfit(x, -window_zz[ii], 1)[0] / (111e3 * coslat)
                sy_sg[i, j] = np.polyfit(y, -window_zz[ii], 1)[0] / 111e3
                '''

            Pdia_sg[i, j], Pdia_sg_lo[i, j], Pdia_sg_hi[i, j], Pdia_sg_boot_all[i,j,:] = bootstrap_Pdia(
            window_epsilon[ii], dCTdz_sg[i,j], N2_sg[i,j], gammamix=GAMMA_mix, n_boot=n_boot_i, n_eff=n_eff, return_bootstrap=True, rng=rng)   
        
            PdiaS_sg[i, j], PdiaS_sg_lo[i, j], PdiaS_sg_hi[i, j], PdiaS_sg_boot_all[i,j,:] = bootstrap_Pdia(
            window_epsilon[ii], dSAdz_sg[i,j], N2_sg[i,j], gammamix=GAMMA_mix, n_boot=n_boot_i, n_eff=n_eff, return_bootstrap=True, rng=rng)    
        
            Pdia_poly_sg[i, j], Pdia_poly_sg_lo[i, j], Pdia_poly_sg_hi[i, j], Pdia_poly_boot = bootstrap_CI_var(window_Pdia[ii], n_boot=n_boot_i, n_eff=n_eff, return_bootstrap=True, rng=rng)    
            PdiaS_poly_sg[i, j], PdiaS_poly_sg_lo[i, j], PdiaS_poly_sg_hi[i, j], PdiaS_poly_boot = bootstrap_CI_var(window_PdiaS[ii], n_boot=n_boot_i, n_eff=n_eff, return_bootstrap=True, rng=rng)
        
            Piso_poly_boot = chi_sg_boot_all[i,j,:] - Pdia_poly_boot
            Piso_poly_sg[i, j] = np.mean(Piso_poly_boot)
            Piso_poly_sg_lo[i, j] = np.percentile(Piso_poly_boot, 2.5)
            Piso_poly_sg_hi[i, j] = np.percentile(Piso_poly_boot, 97.5)
    
            PSiso_poly_boot = chi_sg_boot_all[i,j,:] - PdiaS_poly_boot
            PSiso_poly_sg[i, j] = np.mean(PSiso_poly_boot)
            PSiso_poly_sg_lo[i, j] = np.percentile(PSiso_poly_boot, 2.5)
            PSiso_poly_sg_hi[i, j] = np.percentile(PSiso_poly_boot, 97.5)
        
            Krho_sg[i, j], Krho_sg_lo[i, j], Krho_sg_hi[i, j], Krho_sg_boot_all[i,j,:] = bootstrap_Krho(window_epsilon[ii], N2_sg[i,j], gammamix=GAMMA_mix, n_boot = n_boot_i, n_eff=n_eff, return_bootstrap = True, rng=rng)
    Piso_sg_boot_all = chi_sg_boot_all - Pdia_sg_boot_all
    Piso_sg = np.nanmean(Piso_sg_boot_all, axis = 2) #boot mean
    Piso_sg_mean = np.nanmean(Piso_sg_boot_all, axis = (1,2)) #mean profile (boot and time mean)
    Piso_sg_lo = np.percentile(Piso_sg_boot_all, 2.5, axis = (1,2)) #low percentile (boot and time mean)
    Piso_sg_hi = np.percentile(Piso_sg_boot_all, 97.5, axis = (1,2)) #high percentile (boot and time mean)

    PSiso_sg_boot_all = chiS_sg_boot_all - PdiaS_sg_boot_all
    PSiso_sg = np.nanmean(PSiso_sg_boot_all, axis = 2)
    PSiso_sg_lo = np.percentile(PSiso_sg_boot_all, 2.5, axis = (1,2))
    PSiso_sg_hi = np.percentile(PSiso_sg_boot_all, 97.5, axis = (1,2))
    #removes negatives
    Piso_sg[Piso_sg<0] = 1e-12
    Piso_poly_sg[Piso_poly_sg<0] = 1e-12
    PSiso_sg[PSiso_sg<0] = 1e-14
    PSiso_poly_sg[PSiso_poly_sg<0] = 1e-14

    KisoT = (Piso_sg) / (dCTdx_sg**2 + dCTdy_sg**2) / 2
    KisoS = (PSiso_sg) / (dSAdx_sg**2 + dSAdy_sg**2) / 2

    KisoT_poly = Piso_poly_sg / (dCTdx_sg**2 + dCTdy_sg**2) / 2
    KisoS_poly = PSiso_poly_sg / (dSAdx_sg**2 + dSAdy_sg**2) / 2


    #geostropic velocities
    fin = sw.f(np.nanmean(lat))
    if global_mode and abs(fin) < float(config.get('min_abs_coriolis', 1e-6)):
        fin = np.nan
    dudz = sy_sg * N2_sg / fin
    dvdz = -sx_sg * N2_sg / fin

    ugeos0 = np.full((dudz.shape[0]+1, dudz.shape[1]), 0.)
    vgeos0 = np.full((dudz.shape[0]+1, dudz.shape[1]), 0.)

    for j in range(TIME.size):
        for i in range(SG.size):
            ugeos0[i,j] = np.nansum(dudz[i:,j] * h_sg[i:,j])
            vgeos0[i,j] = np.nansum(dvdz[i:,j] * h_sg[i:,j])
    ugeos = 0.5 * ugeos0[1:] + 0.5 * ugeos0[:-1,:]
    vgeos = 0.5 * vgeos0[1:] + 0.5 * vgeos0[:-1,:]


    #sets level of no motion
    SG_nomo = float(config.get('reference_density', 28.0))
    reference_indices = np.where(SG <= SG_nomo)[0]
    reference_covered = reference_indices.size and (not global_mode or SG[0] <= SG_nomo <= SG[-1])
    if reference_covered:
        jnomo = reference_indices[-1]
        unomo = ugeos[jnomo, :].copy()
        vnomo = vgeos[jnomo, :].copy()
        ugeos -= unomo
        vgeos -= vnomo
        if global_mode:
            reference_valid = np.isfinite(dudz[jnomo, :]) & np.isfinite(dvdz[jnomo, :])
            ugeos[:, ~reference_valid] = np.nan
            vgeos[:, ~reference_valid] = np.nan
    else:
        ugeos[:] = np.nan
        vgeos[:] = np.nan

    ugeos[np.isnan(dudz)] = np.nan
    vgeos[np.isnan(dudz)] = np.nan

    hgrCT = (dCTdx_sg)**2 + (dCTdy_sg)**2
    hgrSA = (dSAdx_sg)**2 + (dSAdy_sg)**2
    hgrCT_err = np.sqrt((2 * dCTdx_sg * dCTdx_sg_err)**2 + (2 * dCTdy_sg * dCTdy_sg_err)**2)
    hgrSA_err = np.sqrt((2 * dSAdx_sg * dSAdx_sg_err)**2 + (2 * dSAdy_sg * dSAdy_sg_err)**2)


    #mixing length
    Lmixing_sg = var_SA_sg**.5 / hgrSA**.5

    #converts dates
    if not SEASONAL_MODE:
        DATES = np.array([np.datetime64(time_origin)+np.timedelta64(time0.astype(int),"D") for time0 in TIME])
    else:
        DATES = np.copy(TIME)
    ds = xr.Dataset(
        data_vars = {
            "date": ( ("TIME"), DATES, {"Description": "Dates in datetime64 or DOY for seasonal average", "Units": ""} ),
            "z" : ( ("SG","TIME"), z_sg, {"Description": "Mean isopycnal depth", "Units": "m"} ),
            "h" : ( ("SG","TIME"), h_sg, {"Description": "Mean isopycnal thickness", "Units": "m"} ),
            "CT" : ( ("SG","TIME"), CT_sg, {"Description": "Conservative Temperature", "Units": "degC"} ),
            "SA" : ( ("SG","TIME"), SA_sg, {"Description": "Salinity Absolute", "Units": "g/kg"} ),
            "dCTdz" : ( ("SG","TIME"), dCTdz_sg, {"Description": "Conservative Temperature vertical gradient", "Units": "degC/m"} ),
            "dCTdx" : ( ("SG","TIME"), dCTdx_sg, {"Description": "Conservative Temperature zonal gradient", "Units": "degC/m"} ),
            "dCTdy" : ( ("SG","TIME"), dCTdy_sg, {"Description": "Conservative Temperature meridional gradient", "Units": "degC/m"} ),
            "dSAdz" : ( ("SG","TIME"), dSAdz_sg, {"Description": "Absolute salinity vertical gradient", "Units": "g/kg/m"} ),
            "dSAdx" : ( ("SG","TIME"), dSAdx_sg, {"Description": "Absolute salinity zonal gradient", "Units": "g/kg/m"} ),
            "dSAdy" : ( ("SG","TIME"), dSAdy_sg, {"Description": "Absolute salinity meridional gradient", "Units": "g/kg/m"} ),
            "hgrCT" : ( ("SG","TIME"), hgrCT, {"Description": "Conservative temperature horizontal squared gradient", "Units": "degC^2/m^2"} ),
            "hgrSA" : ( ("SG","TIME"), hgrSA, {"Description": "Absolute salinity horizontal squared gradient", "Units": "g^2/kg^2/m^2"} ),
            "Lmixing_sg" : ( ("SG","TIME"), Lmixing_sg, {"Description": "Mixing length", "Units": "m"} ),
            "iso_slope_x" : ( ("SG","TIME"), sx_sg, {"Description": "Zonal isopycnal slope", "Units": ""} ),
            "iso_slope_y" : ( ("SG","TIME"), sy_sg, {"Description": "Meridional isopycnal slope", "Units": ""} ),
            "N2" : ( ("SG","TIME"), N2_sg, {"Description": "Buoyancy frequency squared", "Units": "1/s^2"} ),

            "epsilon"     : ( ("SG","TIME"), epsilon_sg, {"Description": "TKE dissipation", "Units": "W/kg"} ),
            "epsilon_lo"  : ( ("SG","TIME"), epsilon_sg_lo, {"Description": "Lower 95% CI of TKE dissipation", "Units": "W/kg"} ),
            "epsilon_hi"  : ( ("SG","TIME"), epsilon_sg_hi, {"Description": "Upper 95% CI of TKE dissipation", "Units": "W/kg"} ),

            "Krho"        : ( ("SG","TIME"), Krho_sg, {"Description": "Diapycnal diffusivity", "Units": "m^2/s"} ),
            "Krho_lo"        : ( ("SG","TIME"), Krho_sg_lo, {"Description": "Lower 95% CI of Diapycnal diffusivity", "Units": "m^2/s"} ),
            "Krho_hi"        : ( ("SG","TIME"), Krho_sg_hi, {"Description": "Lower 95% CI of Diapycnal diffusivity", "Units": "m^2/s"} ),

            "chi"         : ( ("SG","TIME"), chi_sg, {"Description": "Thermal variance dissipation", "Units": "K^2/s"} ),
            "chi_lo"      : ( ("SG","TIME"), chi_sg_lo, {"Description": "Lower 95% CI of chi", "Units": "K^2/s"} ),
            "chi_hi"      : ( ("SG","TIME"), chi_sg_hi, {"Description": "Upper 95% CI of chi", "Units": "K^2/s"} ),

            "Pdia"        : ( ("SG","TIME"), Pdia_sg, {"Description": "Diapycnal production of thermal variance", "Units": "K^2/s"} ),
            "Pdia_lo"     : ( ("SG","TIME"), Pdia_sg_lo, {"Description": "Lower 95% CI of Pdia", "Units": "K^2/s"} ),
            "Pdia_hi"     : ( ("SG","TIME"), Pdia_sg_hi, {"Description": "Upper 95% CI of Pdia", "Units": "K^2/s"} ),

            "Pdia_poly"   : ( ("SG","TIME"), Pdia_poly_sg, {"Description": "Pdia from polynomial fits", "Units": "K^2/s"} ),
            "Pdia_poly_lo": ( ("SG","TIME"), Pdia_poly_sg_lo, {"Description": "Lower 95% CI of Pdia_poly", "Units": "K^2/s"} ),
            "Pdia_poly_hi": ( ("SG","TIME"), Pdia_poly_sg_hi, {"Description": "Upper 95% CI of Pdia_poly", "Units": "K^2/s"} ),

            "Piso"        : ( ("SG","TIME"), Piso_sg, {"Description": "Isopycnal production of thermal variance", "Units": "K^2/s"} ),
            "Piso_poly"   : ( ("SG","TIME"), Piso_poly_sg, {"Description": "Isopycnal production from polynomial chi - Pdia_poly", "Units": "K^2/s"} ),

            "chiS"        : ( ("SG","TIME"), chiS_sg, {"Description": "Haline variance dissipation", "Units": "PSU^2/s"} ),
            "chiS_lo"     : ( ("SG","TIME"), chiS_sg_lo, {"Description": "Lower 95% CI of chiS", "Units": "PSU^2/s"} ),
            "chiS_hi"     : ( ("SG","TIME"), chiS_sg_hi, {"Description": "Upper 95% CI of chiS", "Units": "PSU^2/s"} ),

            "PSdia"       : ( ("SG","TIME"), PdiaS_sg, {"Description": "Diapycnal production of haline variance", "Units": "PSU^2/s"} ),
            "PSdia_lo"    : ( ("SG","TIME"), PdiaS_sg_lo, {"Description": "Lower 95% CI of PSdia", "Units": "PSU^2/s"} ),
            "PSdia_hi"    : ( ("SG","TIME"), PdiaS_sg_hi, {"Description": "Upper 95% CI of PSdia", "Units": "PSU^2/s"} ),

            "PSdia_poly"  : ( ("SG","TIME"), PdiaS_poly_sg, {"Description": "PSdia from polynomial fits", "Units": "PSU^2/s"} ),
            "PSdia_poly_lo":( ("SG","TIME"), PdiaS_poly_sg_lo, {"Description": "Lower 95% CI of PSdia_poly", "Units": "PSU^2/s"} ),
            "PSdia_poly_hi":( ("SG","TIME"), PdiaS_poly_sg_hi, {"Description": "Upper 95% CI of PSdia_poly", "Units": "PSU^2/s"} ),

            "PSiso"       : ( ("SG","TIME"), PSiso_sg, {"Description": "Isopycnal production of haline variance", "Units": "PSU^2/s"} ),
            "PSiso_poly"  : ( ("SG","TIME"), PSiso_poly_sg, {"Description": "Isopycnal production from polynomial chiS - PSdia_poly", "Units": "PSU^2/s"} ),

            "KisoT"       : ( ("SG","TIME"), KisoT, {"Description": "Isopycnal diffusivity from theta", "Units": "m^2/s"} ),
            "KisoS"       : ( ("SG","TIME"), KisoS, {"Description": "Isopycnal diffusivity from salinity", "Units": "m^2/s"} ),
            "KisoT_poly"  : ( ("SG","TIME"), KisoT_poly, {"Description": "Isopycnal diffusivity from theta using polynomial fit", "Units": "m^2/s"} ),
            "KisoS_poly"  : ( ("SG","TIME"), KisoS_poly, {"Description": "Isopycnal diffusivity from salinity using polynomial fit", "Units": "m^2/s"} ),
        },
        coords = {
            "SG": SG,
            "TIME": TIME,
        },  
        attrs = {
            "Description": "Regional fine-scale mixing with Argo in neutral density coordinates"
        },
    )
    ds.SG.attrs['Description'] = 'Neutral density coordinate (kg/m³)'
    ds.TIME.attrs['Description'] = f'Days since {time_origin}, or DOY for seasonal average'



    # Additional diagnostics already computed in the original notebook, now kept.
    for name, values, units, description in [
        ('pressure', p_sg, 'dbar', 'Mean pressure in the neutral-density layer'),
        ('var_CT', var_CT_sg, 'degC^2', 'Variance of CT minus polynomial background'),
        ('var_SA', var_SA_sg, '(g/kg)^2', 'Variance of SA minus polynomial background'),
        ('dCTdx_stderr', dCTdx_sg_err, 'degC/m', 'Zonal CT regression slope standard error'),
        ('dCTdy_stderr', dCTdy_sg_err, 'degC/m', 'Meridional CT regression slope standard error'),
        ('dSAdx_stderr', dSAdx_sg_err, 'g/kg/m', 'Zonal SA regression slope standard error'),
        ('dSAdy_stderr', dSAdy_sg_err, 'g/kg/m', 'Meridional SA regression slope standard error'),
        ('dCTdx_pvalue', dCTdx_sg_p, '1', 'Zonal CT regression p value'),
        ('dCTdy_pvalue', dCTdy_sg_p, '1', 'Meridional CT regression p value'),
        ('dSAdx_pvalue', dSAdx_sg_p, '1', 'Zonal SA regression p value'),
        ('dSAdy_pvalue', dSAdy_sg_p, '1', 'Meridional SA regression p value'),
        ('iso_slope_x_stderr', sx_sg_err, '1', 'Zonal isopycnal slope standard error'),
        ('iso_slope_y_stderr', sy_sg_err, '1', 'Meridional isopycnal slope standard error'),
        ('iso_slope_x_pvalue', sx_sg_p, '1', 'Zonal isopycnal slope p value'),
        ('iso_slope_y_pvalue', sy_sg_p, '1', 'Meridional isopycnal slope p value'),
        ('iso_slope_x_tstat', sx_t_stat, '1', 'Zonal isopycnal slope t statistic'),
        ('iso_slope_x_manual_pvalue', sx_pval, '1', 'Manually evaluated zonal slope p value'),
        ('hgrCT_stderr', hgrCT_err, 'degC^2/m^2', 'Squared CT horizontal gradient error'),
        ('hgrSA_stderr', hgrSA_err, '(g/kg)^2/m^2', 'Squared SA horizontal gradient error'),
        ('ugeos', ugeos, 'm/s', 'Zonal geostrophic velocity relative to reference density'),
        ('vgeos', vgeos, 'm/s', 'Meridional geostrophic velocity relative to reference density'),
        ('dudz', dudz, '1/s', 'Zonal geostrophic vertical shear'),
        ('dvdz', dvdz, '1/s', 'Meridional geostrophic vertical shear'),
        ('Piso_poly_lo', Piso_poly_sg_lo, 'K^2/s', 'Lower 95% CI of polynomial isopycnal production'),
        ('Piso_poly_hi', Piso_poly_sg_hi, 'K^2/s', 'Upper 95% CI of polynomial isopycnal production'),
        ('PSiso_poly_lo', PSiso_poly_sg_lo, 'PSU^2/s', 'Lower 95% CI of legacy polynomial haline production'),
        ('PSiso_poly_hi', PSiso_poly_sg_hi, 'PSU^2/s', 'Upper 95% CI of legacy polynomial haline production'),
        ('profile_count_bin', Nprof_bin_sg, '1', 'Profiles with valid chi and chiS in each density/time window'),
        ('observation_count_bin', Nobs_bin_sg, '1', 'Depth samples with valid chi and chiS in each density/time window'),
        ('effective_sample_count', N_eff_sg, '1', 'Legacy effective sample count used for bootstrap'),
        ('regression_sample_count', n, '1', 'Depth samples used in horizontal regressions'),
    ]:
        ds[name] = xr.DataArray(values, dims=('SG', 'TIME'), attrs={'Units': units, 'Description': description})
    for name, values, units, description in [
        ('profile_count', Nprof_sg, '1', 'Profiles in each time window before mixing-data validity selection'),
        ('latitude', lat_sg, 'degrees_north', 'Mean profile latitude in each time window'),
        ('longitude', lon_sg, 'degrees_east', 'Mean profile longitude in each time window; local unwrap in global mode'),
    ]:
        ds[name] = xr.DataArray(values, dims=('TIME',), attrs={'Units': units, 'Description': description})
    # Preserve the notebook's across-time-and-bootstrap CI convention explicitly.
    for name, values in [('Piso_profile_lo', Piso_sg_lo), ('Piso_profile_hi', Piso_sg_hi),
                         ('PSiso_profile_lo', PSiso_sg_lo), ('PSiso_profile_hi', PSiso_sg_hi)]:
        ds[name] = xr.DataArray(values, dims=('SG',), attrs={'Description': 'Legacy percentile over pooled time and bootstrap dimensions'})
    if config.get('include_bootstrap', False):
        for name, values in [('chi', chi_sg_boot_all), ('chiS', chiS_sg_boot_all),
                             ('Pdia', Pdia_sg_boot_all), ('PSdia', PdiaS_sg_boot_all),
                             ('Piso', Piso_sg_boot_all), ('PSiso', PSiso_sg_boot_all),
                             ('Krho', Krho_sg_boot_all)]:
            ds[name + '_bootstrap'] = xr.DataArray(values, dims=('SG', 'TIME', 'bootstrap'))
    ds.attrs.update({
        'source_notebook': 'plots_mixing_NATRE_gamman_analysis.ipynb cells 2-4, 7-8, 12-16, 18',
        'legacy_conventions': 'depthwise 3-sigma log-epsilon upper clipping; log-depth interpolation; overlapping density windows +/- density_step; fixed-N2/gradient epsilon bootstrap; L_eps=4m unless configured; PSiso_poly uses temperature chi; negative production floors 1e-12/1e-14; independent bootstrap draws; pooled time/bootstrap percentiles; legacy regression and reference-density formulas',
        'structural_repairs': 'bootstrap_Krho invalid-data branch returns four NaN entries when requested; missing reference-density bins yield NaN velocity',
        'analysis_config': json.dumps(config, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)),
        'time_origin': time_origin,
        'seasonal_mode': int(SEASONAL_MODE),
        'bootstrap_count': n_boot_i,
        'bootstrap_seed': 'unseeded' if seed is None else str(seed),
        'density_half_width': dSG,
        'mixing_efficiency': GAMMA_mix,
        'vertical_correlation_length_m': L_eps,
        'vertical_spacing_m': dz,
        'averaging_window_months': nmonths_window,
        'time_step_months': nmonths,
        'reference_density': SG_nomo,
        'global_geometry_guards': int(global_mode),
        'input_profile_count': int(time.size),
        'rejected_finescale_outlier_count': int(np.count_nonzero(baddata)),
    })
    ds = add_plot_products(ds, {
        'chi': chi_sg_boot_all,
        'chiS': chiS_sg_boot_all,
        'Pdia': Pdia_sg_boot_all,
        'PSdia': PdiaS_sg_boot_all,
        'Piso': Piso_sg_boot_all,
        'PSiso': PSiso_sg_boot_all,
        'Krho': Krho_sg_boot_all,
    })
    return add_quicklook_products(ds, time, lon, lat, z, Z, mean_res,
                                  chi_missing_before_clip, epsilon0, baddata,
                                  epsilon, Krho, GAMMA_mix)


"""Strict, identity-aware comparisons for the NATRE validation gate.

These helpers never treat a matching intersection as full coverage parity. Metadata
and additional diagnostic variables are allowed; every reference science variable
and every reference coordinate must agree within the explicitly recorded tolerance.
"""

from collections import defaultdict, deque
import numpy as np


SUMMARY_DETERMINISTIC_VARIABLES = frozenset({
    "date", "z", "h", "CT", "SA", "dCTdz", "dCTdx", "dCTdy",
    "dSAdz", "dSAdx", "dSAdy", "hgrCT", "hgrSA", "Lmixing_sg",
    "iso_slope_x", "iso_slope_y", "N2",
})


def _comparison_array(actual, expected, rtol, atol):
    """Compare arrays without masking finite/NaN/Inf disagreements."""
    av, ev = np.asarray(actual), np.asarray(expected)
    if av.shape != ev.shape:
        return {"passed": False, "actual_shape": list(av.shape),
                "expected_shape": list(ev.shape), "reason": "shape_mismatch"}
    result = {"rtol": float(rtol), "atol": float(atol), "elements": int(ev.size)}
    if np.issubdtype(av.dtype, np.number) and np.issubdtype(ev.dtype, np.number):
        # Scientific arrays here are real. Retain integer equality when tolerances
        # are zero and do not confuse infinities with missing observations.
        equal = np.isclose(av, ev, rtol=rtol, atol=atol, equal_nan=True)
        finite = np.isfinite(av) & np.isfinite(ev)
        result["finite_mask_mismatches"] = int(np.count_nonzero(np.isfinite(av) != np.isfinite(ev)))
        result["nan_mask_mismatches"] = int(np.count_nonzero(np.isnan(av) != np.isnan(ev)))
        result["finite_pairs"] = int(np.count_nonzero(finite))
        if np.any(finite):
            diff = np.abs(av[finite].astype(np.float64) - ev[finite].astype(np.float64))
            result["max_abs_error"] = float(np.max(diff))
            nonzero = ev[finite] != 0
            result["max_relative_error_nonzero"] = (
                float(np.max(diff[nonzero] / np.abs(ev[finite][nonzero])))
                if np.any(nonzero) else None
            )
        else:
            result["max_abs_error"] = None
            result["max_relative_error_nonzero"] = None
    elif np.issubdtype(av.dtype, np.datetime64) and np.issubdtype(ev.dtype, np.datetime64):
        equal = (av == ev) | (np.isnat(av) & np.isnat(ev))
    elif np.issubdtype(av.dtype, np.timedelta64) and np.issubdtype(ev.dtype, np.timedelta64):
        equal = (av == ev) | (np.isnat(av) & np.isnat(ev))
    else:
        equal = av == ev
    unequal = ~equal
    result["mismatched_elements"] = int(np.count_nonzero(unequal))
    result["passed"] = not bool(np.any(unequal))
    if not result["passed"]:
        first = tuple(int(v) for v in np.argwhere(unequal)[0])
        result["first_mismatch"] = {
            "index": list(first), "actual": str(av[first]), "expected": str(ev[first])
        }
    return result


def _comparison_profile_keys(ds):
    """Position and time are the identities retained by the old output format."""
    required = ("time", "longitude", "latitude")
    missing = [name for name in required if name not in ds]
    if missing:
        raise ValueError("Missing legacy profile identity fields: " + ", ".join(missing))
    times = np.asarray(ds.time.values).astype("datetime64[ns]").astype(np.int64)
    longitude, latitude = np.asarray(ds.longitude.values), np.asarray(ds.latitude.values)
    if not (times.ndim == longitude.ndim == latitude.ndim == 1):
        raise ValueError("time, longitude, and latitude must be one-dimensional")

    def position(value):
        # NaNs must be hashable/comparable for accounting, not collapsed profiles.
        return "NaN" if np.isnan(value) else float(value)

    return [(int(t), position(x), position(y)) for t, x, y in zip(times, longitude, latitude)]


def _comparison_key_display(key):
    return {"time": str(np.datetime64(key[0], "ns")), "longitude": key[1], "latitude": key[2]}


def _comparison_variables(actual, expected, rtol, atol, tolerances=None):
    """Compare one variable at a time to avoid a second whole-dataset copy."""
    tolerances = tolerances or {}
    missing = sorted(set(expected.data_vars) - set(actual.data_vars))
    variables = {}
    for name in expected.data_vars:
        if name in missing:
            variables[name] = {"passed": False, "reason": "missing_variable"}
            continue
        av, ev = actual[name], expected[name]
        if set(av.dims) != set(ev.dims):
            variables[name] = {"passed": False, "reason": "dimension_mismatch",
                               "actual_dims": list(av.dims), "expected_dims": list(ev.dims)}
            continue
        av = av.transpose(*ev.dims)
        vrtol, vatol = tolerances.get(name, (rtol, atol))
        variables[name] = _comparison_array(av.values, ev.values, vrtol, vatol)
    return variables, missing


def compare_profile_datasets(actual, expected, rtol=1e-10, atol=1e-12, tolerances=None):
    """Return JSON-serializable validation; profile order may differ.

    Match exact time/longitude/latitude, retaining the occurrence count for duplicate
    keys. Ambiguous identical-key profiles retain their relative order. Coverage is
    always part of ``passed``. If coverage differs, array diagnostics cover matching
    occurrences only and are labelled explicitly. For tiny mixing diagnostics, use
    ``atol=0`` (recommended) or pass variable-specific ``(rtol, atol)`` tolerances.
    """
    ak, ek = _comparison_profile_keys(actual), _comparison_profile_keys(expected)
    available = defaultdict(deque)
    for index, key in enumerate(ak):
        available[key].append(index)
    duplicate_actual = sum(len(indices) > 1 for indices in available.values())
    seen_expected = defaultdict(int)
    ai, ei, missing = [], [], []
    for index, key in enumerate(ek):
        seen_expected[key] += 1
        if available[key]:
            ai.append(available[key].popleft())
            ei.append(index)
        else:
            missing.append(key)
    extra = [key for key, indices in available.items() for _ in indices]
    coverage = {
        "passed": not missing and not extra,
        "actual_profiles": len(ak), "expected_profiles": len(ek),
        "matched_profiles": len(ai), "missing_profiles": len(missing),
        "extra_profiles": len(extra),
        "duplicate_actual_identity_keys": duplicate_actual,
        "duplicate_expected_identity_keys": sum(count > 1 for count in seen_expected.values()),
        "identity": "exact (time_ns, longitude, latitude), duplicate occurrences retained",
        "missing_examples": [_comparison_key_display(k) for k in missing[:10]],
        "extra_examples": [_comparison_key_display(k) for k in extra[:10]],
    }
    aa = actual.isel(time=ai)
    ee = expected.isel(time=ei)
    coordinates = {}
    for name in expected.coords:
        if name not in actual:
            coordinates[name] = {"passed": False, "reason": "missing_coordinate"}
        else:
            coordinates[name] = _comparison_array(aa[name].values, ee[name].values, 0, 0)
    variables, missing_variables = _comparison_variables(aa, ee, rtol, atol, tolerances)
    passed = coverage["passed"] and all(v["passed"] for v in coordinates.values()) and all(v["passed"] for v in variables.values())
    return {
        "passed": bool(passed), "comparison": "legacy_profile_metrics",
        "array_comparison_scope": "all_profiles" if coverage["passed"] else "matched_profiles_only",
        "coverage": coverage, "coordinates": coordinates, "variables": variables,
        "missing_variables": missing_variables,
        "additional_variables": sorted(set(actual.data_vars) - set(expected.data_vars)),
        "failed_variables": [k for k, v in variables.items() if not v["passed"]],
    }


def compare_summary_datasets(actual, expected, rtol=1e-10, atol=1e-12, tolerances=None):
    """Strict comparison with separate deterministic/bootstrap diagnostics.

    The archived notebook did not store its random seed. Its bootstrap-derived
    means, intervals, residuals and diffusivities will generally differ on rerun.
    This helper reports those differences and NEVER exempts them from ``passed``.
    Use an independently replayed legacy notebook with the same fixed random seed
    to establish algorithm parity for those fields.
    """
    coordinates = {}
    for name in expected.coords:
        coordinates[name] = (
            _comparison_array(actual[name].values, expected[name].values, 0, 0)
            if name in actual else {"passed": False, "reason": "missing_coordinate"}
        )
    variables, missing_variables = _comparison_variables(actual, expected, rtol, atol, tolerances)
    deterministic = {k: v for k, v in variables.items() if k in SUMMARY_DETERMINISTIC_VARIABLES}
    stochastic = {k: v for k, v in variables.items() if k not in SUMMARY_DETERMINISTIC_VARIABLES}
    coordinates_passed = all(v["passed"] for v in coordinates.values())
    return {
        "passed": bool(coordinates_passed and all(v["passed"] for v in variables.values())),
        "comparison": "legacy_regional_summary", "coordinates": coordinates,
        "variables": variables, "missing_variables": missing_variables,
        "additional_variables": sorted(set(actual.data_vars) - set(expected.data_vars)),
        "deterministic_passed": bool(coordinates_passed and all(v["passed"] for v in deterministic.values())),
        "bootstrap_derived_passed": bool(coordinates_passed and all(v["passed"] for v in stochastic.values())),
        "deterministic_variables": sorted(deterministic),
        "bootstrap_derived_variables": sorted(stochastic),
        "failed_variables": [k for k, v in variables.items() if not v["passed"]],
        "bootstrap_note": "Archived notebook results are unseeded; only a same-seed independent legacy replay can establish reproducible bootstrap parity.",
    }


def _json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def _data_directory(path):
    """All generated data, manifests and Dask spill belong on the data volume."""
    path = Path(path).expanduser().resolve()
    base = Path('/swot/SUM05/amf2288').resolve()
    if not path.is_relative_to(base):
        raise ValueError(f'Output must be below {base}; refusing {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _signature(config):
    # Scheduling choices do not affect numerical results or invalidate a restart.
    keep = {k: v for k, v in config.items()
            if k not in {'workers', 'memory_limit', 'output', 'validation_report'}}
    return _fingerprint(keep)


def _source_snapshot(config):
    if config['source'] == 'local':
        paths = [Path(config['gdac']) / 'ar_index_global_prof.txt']
    else:
        paths = sorted(Path(config['reference_dir']).glob('NATRE_????_????_Argo_resmode.nc'))
    return [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            for p in paths]


def _inside_region(lon, lat, bounds):
    west, east, south, north = bounds
    longitude = ((lon + 180) % 360) - 180
    if west == -180 and east == 180:
        in_lon = np.ones_like(longitude, dtype=bool)
    elif west < east:
        in_lon = (longitude >= west) & (longitude < east)
    else:
        in_lon = (longitude >= west) | (longitude < east)
    in_lat = (lat >= south) & ((lat <= north) if north == 90 else (lat < north))
    return in_lon & in_lat


def _region_id(lon, lat, cell_size):
    if cell_size is None:
        return 'region'
    x = min(int((float(lon) + 180) % 360 // cell_size), int(np.ceil(360 / cell_size)) - 1)
    y = min(int((float(lat) + 90) // cell_size), int(np.ceil(180 / cell_size)) - 1)
    return f'x{x:03d}_y{y:03d}'


def iter_local_index(config, inventory=None):
    """Stream the local core index; never construct a remote fetcher."""
    import pandas as pd
    root = Path(config['gdac']).resolve()
    index = root / 'ar_index_global_prof.txt'
    if not root.is_dir() or not (root / 'dac').is_dir() or not index.is_file():
        raise ValueError(f'Not a local GDAC archive: {root}')
    state = inventory if inventory is not None else {}
    state.update(index=str(index), index_bytes=index.stat().st_size,
                 index_mtime_ns=index.stat().st_mtime_ns, selected_files=0,
                 first_profile=None, last_profile=None, years={})
    with index.open() as stream:
        state['index_header'] = [next(stream).rstrip() for _ in range(8)]
    # Keep the small duplicate set on disk only if needed; the GDAC index itself
    # guarantees one row per mono-profile filename. Detect duplicates in each batch
    # and also reject duplicate scientific identities when comparing full outputs.
    for frame in pd.read_csv(index, comment='#', dtype=str, chunksize=100_000):
        dates = pd.to_datetime(frame['date'], format='%Y%m%d%H%M%S', errors='coerce')
        lon = pd.to_numeric(frame['longitude'], errors='coerce').to_numpy()
        lat = pd.to_numeric(frame['latitude'], errors='coerce').to_numpy()
        mask = _inside_region(lon, lat, config['bounds']) & dates.notna().to_numpy()
        if config.get('start'):
            mask &= (dates >= pd.Timestamp(config['start'])).to_numpy()
        if config.get('end'):
            mask &= (dates < pd.Timestamp(config['end'])).to_numpy()
        for row in frame.loc[mask].itertuples(index=False):
            rel = str(row.file)
            file_path = (root / 'dac' / rel).resolve()
            if not file_path.is_relative_to(root / 'dac'):
                raise ValueError(f'Index path escapes archive: {rel}')
            date = pd.Timestamp(str(row.date)).isoformat() if '-' in str(row.date) else pd.to_datetime(str(row.date), format='%Y%m%d%H%M%S').isoformat()
            state['selected_files'] += 1
            state['first_profile'] = min(state['first_profile'] or date, date)
            state['last_profile'] = max(state['last_profile'] or date, date)
            year = date[:4]
            state['years'][year] = state['years'].get(year, 0) + 1
            yield {'path': str(file_path), 'source_file': rel, 'date': date,
                   'longitude': float(row.longitude), 'latitude': float(row.latitude)}


def _load_local_records(records, config):
    """Use the installed Argopy GDAC preprocessing and research QC on local files."""
    import argopy
    from argopy.data_fetchers.gdac_data_processors import pre_process_multiprof
    profiles = []
    rejected = []
    original_profiles = 0
    for record in records:
        with xr.open_dataset(record['path'], engine='netcdf4') as opened:
            # Argopy parses this unused history field without accepting blanks.
            # It is discarded by profile2point anyway; retain all scientific/QC fields.
            raw = opened.drop_vars('SCIENTIFIC_CALIB_DATE', errors='ignore').load()
        original_profiles += raw.sizes.get('N_PROF', 0)
        points = pre_process_multiprof(raw, access_point='BOX', access_point_opts={})
        points = points.argo.filter_researchmode()
        if points.sizes.get('N_POINTS', 0) == 0:
            rejected.append({'source_file': record['source_file'], 'reason': 'research_qc'})
            continue
        mask = _inside_region(points.LONGITUDE, points.LATITUDE, config['bounds'])
        mask &= (points.PRES >= 0) & (points.PRES < config['max_pressure'])
        if config.get('start'):
            mask &= points.TIME >= np.datetime64(config['start'])
        if config.get('end'):
            mask &= points.TIME < np.datetime64(config['end'])
        points = points.where(mask, drop=True)
        if points.sizes.get('N_POINTS', 0) == 0:
            rejected.append({'source_file': record['source_file'], 'reason': 'selection_after_qc'})
            continue
        needed = ['PRES', 'TEMP', 'PSAL', 'TIME', 'LATITUDE', 'LONGITUDE',
                  'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'DIRECTION']
        missing = {'PRES', 'TEMP', 'PSAL'} - set(points.data_vars)
        if missing:
            rejected.append({'source_file': record['source_file'], 'reason': 'missing_' + '_'.join(sorted(missing))})
            continue
        prof = points[[k for k in needed if k in points]].argo.point2profile()
        if prof.sizes.get('N_LEVELS', 0) < 2:
            rejected.append({'source_file': record['source_file'], 'reason': 'fewer_than_two_levels'})
            continue
        for variable in ['PRES', 'TEMP', 'PSAL']:
            if 'N_LEVELS' not in prof[variable].dims:
                prof[variable] = xr.broadcast(prof[variable], prof.N_LEVELS)[0].transpose('N_PROF', 'N_LEVELS')
        prof['SOURCE_FILE'] = ('N_PROF', [record['source_file']] * prof.sizes['N_PROF'])
        profiles.append(prof)
    if not profiles:
        return None, {'raw_profiles': original_profiles, 'rejections': rejected}
    combined = xr.concat(profiles, dim='N_PROF', join='outer', combine_attrs='drop_conflicts')
    combined = combined.assign_coords(N_PROF=np.arange(combined.sizes['N_PROF']))
    return combined, {'raw_profiles': original_profiles, 'rejections': rejected,
                      'argopy_path': argopy.__file__}


def _profile_tasks(config, inventory):
    size = config['batch_size']
    if config['source'] == 'legacy_raw':
        reference = Path(config['reference_dir'])
        paths = sorted(reference.glob('NATRE_????_????_Argo_resmode.nc'))
        for path in paths:
            with xr.open_dataset(path) as ds:
                keep = np.ones(ds.sizes['N_PROF'], dtype=bool)
                if config.get('start'):
                    keep &= ds.TIME.values >= np.datetime64(config['start'])
                if config.get('end'):
                    keep &= ds.TIME.values < np.datetime64(config['end'])
                keep &= _inside_region(ds.LONGITUDE.values, ds.LATITUDE.values, config['bounds'])
                indices = np.flatnonzero(keep)
            inventory['selected_files'] = inventory.get('selected_files', 0) + 1
            inventory['selected_raw_profiles'] = inventory.get('selected_raw_profiles', 0) + len(indices)
            for offset in range(0, len(indices), size):
                yield {'id': f'{path.stem}_{offset:06d}', 'region': 'region',
                       'raw_path': str(path), 'indices': indices[offset:offset + size].tolist()}
        return
    buckets, counters = {}, collections.Counter()
    for record in iter_local_index(config, inventory):
        region = _region_id(record['longitude'], record['latitude'], config['cell_size'])
        bucket = buckets.setdefault(region, [])
        bucket.append(record)
        if len(bucket) >= size:
            yield {'id': f'part_{counters[region]:07d}', 'region': region, 'records': bucket}
            counters[region] += 1
            buckets[region] = []
    for region, bucket in sorted(buckets.items()):
        if bucket:
            yield {'id': f'part_{counters[region]:07d}', 'region': region, 'records': bucket}


def _netcdf_write(ds, path):
    temporary = path.with_suffix('.partial.nc')
    encoding = {name: {'zlib': True, 'complevel': 2}
                for name in ds.data_vars if ds[name].dtype.kind in 'fiu'}
    ds.to_netcdf(temporary, engine='netcdf4', encoding=encoding)
    temporary.replace(path)


def _profile_task(task, config, signature):
    directory = Path(config['output']) / 'profiles' / task['region']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (task['id'] + '.nc')
    status_path = directory / (task['id'] + '.json')
    source_paths = [task['raw_path']] if 'raw_path' in task else [r['path'] for r in task['records']]
    source_stats = []
    for source_path in source_paths:
        p = Path(source_path)
        source_stats.append((source_path, p.stat().st_size, p.stat().st_mtime_ns) if p.exists()
                            else (source_path, None, None))
    signature = _fingerprint({'run': signature, 'task': task, 'files': source_stats})
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if (previous.get('signature') == signature and previous.get('status') == 'complete'
                and (previous.get('accepted_profiles') == 0 or path.is_file())):
            return {**previous, 'resumed': True}
    try:
        if 'raw_path' in task:
            with xr.open_dataset(task['raw_path']) as source:
                raw = source.isel(N_PROF=task['indices']).load()
            raw['SOURCE_FILE'] = ('N_PROF', [task['raw_path']] * raw.sizes['N_PROF'])
            accounting = {'raw_profiles': raw.sizes['N_PROF'], 'rejections': []}
        else:
            raw, accounting = _load_local_records(task['records'], config)
        if raw is None:
            accepted = 0
            attributes = {}
        else:
            # Numerical warnings are expected for unsampled levels, not task failures.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = process_profiles(raw, config['profile_config'])
            accepted = result.sizes.get('time', 0)
            attributes = {k: v for k, v in result.attrs.items()
                          if isinstance(v, (int, float, str))}
            if accepted:
                result.attrs.update(run_signature=signature, source_mode=config['source'])
                _netcdf_write(result, path)
        status = {'id': task['id'], 'region': task['region'], 'status': 'complete',
                  'signature': signature, 'path': str(path) if accepted else None,
                  'accepted_profiles': accepted, 'accounting': accounting,
                  'attributes': attributes}
    except Exception:
        status = {'id': task['id'], 'region': task['region'], 'status': 'failed',
                  'signature': signature, 'error': traceback.format_exc()}
    _json_write(status_path, status)
    return status


def _bounded_map(function, items, client, max_pending):
    if client is None:
        for item in items:
            yield function(item)
        return
    from distributed import as_completed
    iterator = iter(items)
    pending = as_completed()
    for _ in range(max_pending):
        try:
            pending.add(client.submit(function, next(iterator), pure=False))
        except StopIteration:
            break
    for future in pending:
        yield future.result()
        future.release()
        try:
            pending.add(client.submit(function, next(iterator), pure=False))
        except StopIteration:
            pass


def _summary_task_impl(item, config):
    region, paths = item
    target = Path(config['output']) / 'summaries' / (region + '.nc')
    target.parent.mkdir(exist_ok=True)
    summary_config = dict(config['summary_config'])
    summary_config['global_mode'] = (config['cell_size'] is not None
                                     or config['bounds'][0] > config['bounds'][1]
                                     or config['bounds'][2] <= 0 <= config['bounds'][3])
    with xr.open_mfdataset(sorted(paths), combine='nested', concat_dim='time',
                           data_vars='minimal', coords='minimal', compat='override') as profiles:
        from dask.utils import parse_bytes
        # The faithful notebook stage materializes one region, including its
        # interpolated fields and temporary masks. Refuse a region too large for
        # its worker instead of letting repeated worker restarts hide the issue.
        estimated_bytes = profiles.sizes['time'] * profiles.sizes['Z'] * 8 * 50
        budget = parse_bytes(config['memory_limit'])
        if estimated_bytes > budget * .65:
            raise MemoryError(f'{region}: summary estimate {estimated_bytes / 2**30:.1f} GiB '
                              f'exceeds 65% of worker budget; use smaller --cell-size '
                              'or increase --memory-limit')
        # Reference notebook used time order. Stable sorting also makes resuming
        # batches and different Dask completion orders immaterial to bootstrapping.
        profiles = profiles.sortby('time')
        origin = np.datetime64(summary_config['time_origin'], 'ns')
        horizon = np.datetime64(config['end'], 'ns') if config.get('end') else profiles.time.max().values.astype('datetime64[ns]')
        max_days = float((horizon - origin) / np.timedelta64(1, 'D'))
        step = 365.25 / 12 * summary_config['nmonths']
        n_windows = max(1, int(np.ceil(max_days / step)))
        summary_config['time_centers_days'] = step * (np.arange(n_windows) + .5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            summary = aggregate_region(profiles, summary_config)
        summary.attrs.update(region_id=region, run_signature=_signature(config),
                             input_profiles=profiles.sizes['time'])
        summary.attrs['longitude_min'] = float(profiles.longitude.min())
        summary.attrs['longitude_max'] = float(profiles.longitude.max())
        summary.attrs['latitude_min'] = float(profiles.latitude.min())
        summary.attrs['latitude_max'] = float(profiles.latitude.max())
        _netcdf_write(summary, target)
    return {'region': region, 'path': str(target), 'status': 'complete'}


def _summary_task(item, config):
    with xr.set_options(file_cache_maxsize=512):
        return _summary_task_impl(item, config)


def run_pipeline(config, summaries=True):
    config = dict(config, source_snapshot=_source_snapshot(config))
    output = _data_directory(config['output'])
    signature = _signature(config)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get('signature') != signature:
            raise ValueError(f'{output} belongs to a different configuration; choose a new output directory')
    manifest = {'signature': signature, 'config': config, 'status': 'running'}
    _json_write(manifest_path, manifest)
    inventory = {}
    results, client, cluster = [], None, None
    try:
        if config['workers'] > 1:
            from distributed import Client, LocalCluster
            spill = output / 'runtime' / 'dask'
            spill.mkdir(parents=True, exist_ok=True)
            cluster = LocalCluster(n_workers=config['workers'], threads_per_worker=1,
                                   processes=True, memory_limit=config['memory_limit'],
                                   local_directory=str(spill), dashboard_address=None)
            client = Client(cluster)
        job = functools.partial(_profile_task, config=config, signature=signature)
        for result in _bounded_map(job, _profile_tasks(config, inventory), client,
                                   2 * config['workers']):
            results.append(result)
            print(f"{config['source']}: {len(results)} batches finished; "
                  f"{sum(r.get('accepted_profiles', 0) for r in results)} accepted profiles", flush=True)
        failed = [r for r in results if r['status'] != 'complete']
        groups = collections.defaultdict(list)
        for result in results:
            if result.get('path'):
                groups[result['region']].append(result['path'])
        _json_write(output / 'inventory.json', inventory)
        manifest.update(inventory=inventory, batches=len(results),
                        accepted_profiles=sum(r.get('accepted_profiles', 0) for r in results),
                        failed_batches=failed, regions=dict(groups))
        if failed:
            raise RuntimeError(f'{len(failed)} processing batches failed; see manifest/status files')
        if not groups:
            raise RuntimeError('No profiles passed selection and quality/resolution criteria')
        if summaries:
            resolved_summary = dict(config['summary_config'])
            if not resolved_summary.get('time_origin'):
                resolved_summary['time_origin'] = inventory['first_profile'][:4] + '-01-01'
            manifest['resolved_summary_config'] = resolved_summary
            summary_job = functools.partial(_summary_task, config=dict(config, summary_config=resolved_summary))
            manifest['summaries'] = list(_bounded_map(summary_job, sorted(groups.items()),
                                                     client, config['workers']))
        manifest['status'] = 'complete'
        return manifest
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc(), inventory=inventory)
        raise
    finally:
        _json_write(manifest_path, manifest)
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()


def _open_profiles(paths):
    return xr.open_mfdataset(sorted(map(str, paths)), combine='nested', concat_dim='time',
                            data_vars='minimal', coords='minimal', compat='override')


def _legacy_summary(profiles, notebook, seed, n_boot):
    """Independent reference execution: original scientific cells, no IO/plot cells."""
    import ast
    import scipy.interpolate as intrp
    import seawater as sw
    from scipy.stats import linregress, t as tdist
    cells = json.loads(Path(notebook).read_text())['cells']
    namespace = {'np': np, 'xr': xr, 'gsw': gsw, 'sw': sw, 'intrp': intrp,
                 'linregress': linregress, 'tdist': tdist, 'ds': profiles.copy(deep=True),
                 'SEASONAL_MODE': False, 'GAMMA_mix': .2, 'nmonths': 3,
                 'nmonths_window': 12., 'print': lambda *a, **k: None}
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        for number in [2, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 18]:
            source = ''.join(cells[number]['source'])
            if number == 14:
                source = source.replace('n_boot_i = 1000', f'n_boot_i = {n_boot}')
            tree = ast.parse(source)
            if number == 5:
                tree.body = [node for node in tree.body
                             if not (isinstance(node, ast.Assign)
                                     and any(isinstance(t, ast.Name) and t.id == 'ds' for t in node.targets))
                             and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                                      and isinstance(node.value.func, ast.Name)
                                      and node.value.func.id == 'display')]
            if number == 18:
                tree.body = [node for node in tree.body if isinstance(node, ast.Assign)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                exec(compile(tree, f'{notebook}:cell{number}', 'exec'), namespace)
        return namespace['ds']
    finally:
        np.random.set_state(state)


def summary_regression(reference_paths, reference_summary, notebook, config, output):
    """Diagnose a failed source comparison without an unnecessary full bootstrap.

    All profiles are used for deterministic fields. A dense-year subset uses the
    full bootstrap count and an independent original-notebook replay. These
    diagnostics alone can NEVER authorize a global run.
    """
    with xr.set_options(file_cache_maxsize=512), _open_profiles(reference_paths) as opened:
        reference = opened.load()
        diagnostic_config = dict(config, n_boot=1)
        print('Checking deterministic summary fields on all reference profiles', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            deterministic = aggregate_region(reference, diagnostic_config)
        with xr.open_dataset(reference_summary) as historical:
            comparison = compare_summary_datasets(deterministic, historical, rtol=0, atol=0)
        deterministic_fields = sorted(SUMMARY_DETERMINISTIC_VARIABLES & set(deterministic.data_vars))
        deterministic = deterministic[deterministic_fields]
        deterministic.attrs['purpose'] = 'Full-population deterministic validation only; bootstrap-dependent variables deliberately omitted'
        _netcdf_write(deterministic, Path(output) / 'reference_deterministic_check.nc')
        years = reference.time.values.astype('datetime64[Y]')
        selected = np.flatnonzero(years == years.min())
        selected = selected[np.linspace(0, len(selected) - 1, min(100, len(selected)), dtype=int)]
        sample = reference.isel(time=selected)
        print(f'Checking {len(selected)} reference profiles with {config["n_boot"]} seeded bootstrap replicates', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            actual = aggregate_region(sample, config)
            expected = _legacy_summary(sample, notebook, config['seed'], config['n_boot'])
        sample_comparison = compare_summary_datasets(actual, expected, rtol=0, atol=0)
        actual.attrs['purpose'] = 'Validation subset, not the complete NATRE regional estimate'
        _netcdf_write(actual, Path(output) / 'seeded_summary_subset.nc')
        return {'full_reference_profiles': reference.sizes['time'],
                'deterministic_passed': comparison['deterministic_passed'],
                'deterministic_variables': {k: comparison['variables'][k] for k in comparison['deterministic_variables']},
                'coordinates': comparison['coordinates'],
                'seeded_subset_profiles': len(selected),
                'seeded_subset_first_time': str(sample.time.min().values),
                'seeded_subset_last_time': str(sample.time.max().values),
                'bootstrap_count': config['n_boot'], 'bootstrap_seed': config['seed'],
                'seeded_subset_comparison': sample_comparison,
                'full_population_bootstrap_comparison_performed': False}


def _validate_natre(config, args):
    """Never promote common-profile agreement to full-archive equivalence."""
    output = _data_directory(config['output'])
    report = {'passed': False, 'global_authorized': False,
              'required_region': [-38, -20, 19, 32],
              'required_time': ['2014-01-01', '2024-01-01'],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gdac': config['gdac'], 'source_snapshot': _source_snapshot(config),
              'summary_config': config['summary_config']}
    _json_write(output / 'validation.json', report)
    expected_paths = sorted(Path(args.reference_dir).glob('Mixing_Argo_NATRE_2m_????_resmode.nc'))
    expected_years = {p.name.split('_')[-2] for p in expected_paths}
    if expected_years != {str(year) for year in range(2014, 2024)}:
        raise ValueError('Validation requires every annual reference from 2014 through 2023')
    try:
        for mode in ['legacy_raw', 'local']:
            stage = dict(config, source=mode, output=str(output / mode),
                         reference_dir=args.reference_dir)
            manifest = run_pipeline(stage, summaries=False)
            paths = [path for group in manifest['regions'].values() for path in group]
            print(f'Comparing all {mode} profile outputs with the saved NATRE reference', flush=True)
            with _open_profiles(paths) as actual, _open_profiles(expected_paths) as expected:
                result = compare_profile_datasets(actual, expected, atol=0)
            report[mode + '_profiles'] = result
            report[mode + '_inventory'] = manifest['inventory']
            _json_write(output / 'validation.json', report)
        if not report['local_profiles']['passed']:
            report['summary_regression'] = summary_regression(
                expected_paths, args.reference_summary, args.legacy_notebook,
                config['summary_config'], output)
            report['full_summary_comparison'] = 'not_run: local profile coverage/values already differ; full-population bootstrap equivalence is not asserted'
            report['status'] = 'mismatch_global_blocked'
            print(json.dumps({'status': report['status'], 'report': str(output / 'validation.json')}, indent=2))
            return report
        # Compare the aggregate implementation on identical full reference inputs.
        # Historical bootstraps were unseeded; they cannot be reproduced bitwise.
        print('Calculating the full seeded regional summary', flush=True)
        with _open_profiles(expected_paths) as expected:
            expected = expected.load()
            summary_config = dict(config['summary_config'])
            actual_summary = aggregate_region(expected, summary_config)
            _netcdf_write(actual_summary, output / 'recomputed_summary.nc')
            print('Replaying original notebook scientific cells with the same seed', flush=True)
            independent = _legacy_summary(expected, args.legacy_notebook,
                                          summary_config['seed'], summary_config['n_boot'])
            _netcdf_write(independent, output / 'seeded_legacy_summary.nc')
            report['seeded_notebook_equivalence'] = compare_summary_datasets(actual_summary, independent, atol=0)
            with xr.open_dataset(args.reference_summary) as historical:
                report['historical_summary'] = compare_summary_datasets(actual_summary, historical, atol=0)
        # The independently seeded reference establishes stochastic algorithm parity;
        # historical deterministic fields must agree and full local input/output parity
        # is mandatory. Do not accept an intersection of the profile populations.
        report['passed'] = bool(report['legacy_raw_profiles']['passed']
                                and report['local_profiles']['passed']
                                and report['seeded_notebook_equivalence']['passed']
                                and report['historical_summary']['deterministic_passed'])
        report['global_authorized'] = report['passed']
        report['status'] = 'passed' if report['passed'] else 'mismatch_global_blocked'
    except Exception:
        report.update(status='error_global_blocked', error=traceback.format_exc())
        raise
    finally:
        _json_write(output / 'validation.json', report)
    print(json.dumps({'status': report['status'], 'report': str(output / 'validation.json')}, indent=2))
    return report


def validate_natre(config, args):
    # A decade produces >128 files. The xarray default would reopen every file
    # while comparing each variable because the previous variable evicted it.
    with xr.set_options(file_cache_maxsize=512):
        return _validate_natre(config, args)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['inventory', 'run', 'validate-natre'])
    parser.add_argument('--gdac', default='/swot/SUM05/dbalwada/Argo_sync')
    parser.add_argument('--bounds', nargs=4, type=float, metavar=('WEST', 'EAST', 'SOUTH', 'NORTH'),
                        default=[-180., 180., -90., 90.])
    parser.add_argument('--start', help='Inclusive UTC date; omitted means earliest locally indexed profile')
    parser.add_argument('--end', help='Exclusive UTC date; omitted means all locally indexed dates')
    parser.add_argument('--output', default='/swot/SUM05/amf2288/argo_mixing/natre_validation')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--memory-limit', default='8GiB')
    parser.add_argument('--cell-size', type=float, default=10., help='Degrees per regional summary cell; 0 means one selected region')
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--density-min', type=float, default=26.75)
    parser.add_argument('--density-max', type=float, default=28.05)
    parser.add_argument('--density-step', type=float, default=.05)
    parser.add_argument('--time-step-months', type=float, default=3.)
    parser.add_argument('--window-months', type=float, default=12.)
    parser.add_argument('--correlation-length', type=float, default=4.)
    parser.add_argument('--reference-dir', default='/swot/SUM05/amf2288/espe_code/data_files')
    parser.add_argument('--reference-summary', default='/swot/SUM05/amf2288/espe_code/NATRE_Argo_mixing_gamman.nc')
    parser.add_argument('--legacy-notebook', default=str(Path(__file__).with_name('plots_mixing_NATRE_gamman_analysis.ipynb')))
    parser.add_argument('--validation-report', help='Passing validate-natre JSON required before run')
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if (args.density_step <= 0 or args.density_min >= args.density_max
            or args.time_step_months <= 0 or args.window_months <= 0
            or args.correlation_length <= 0):
        raise ValueError('Density/time steps, averaging window, and correlation length must be positive; density bounds must increase')
    if args.workers < 1 or args.batch_size < 1 or args.n_boot < 1:
        raise ValueError('workers, batch-size and n-boot must be positive')
    if not 0 <= args.cell_size <= 30:
        raise ValueError('cell-size must be 0 (one region) or <=30 degrees')
    west, east, south, north = args.bounds
    if not (-180 <= west <= 180 and -180 <= east <= 180 and west != east
            and -90 <= south < north <= 90):
        raise ValueError('Invalid longitude/latitude bounds')
    if args.start and args.end and np.datetime64(args.start) >= np.datetime64(args.end):
        raise ValueError('start must precede exclusive end')
    config = {'gdac': str(Path(args.gdac).resolve()), 'bounds': args.bounds,
              'start': args.start, 'end': args.end, 'source': 'local',
              'output': str(Path(args.output).resolve()), 'workers': args.workers,
              'batch_size': args.batch_size, 'memory_limit': args.memory_limit,
              'cell_size': args.cell_size or None, 'max_pressure': 2200.,
              'profile_config': {}, 'summary_config': {'seed': args.seed, 'n_boot': args.n_boot,
                  'density_min': args.density_min, 'density_max': args.density_max,
                  'density_step': args.density_step, 'nmonths': args.time_step_months,
                  'nmonths_window': args.window_months, 'correlation_length_m': args.correlation_length,
                  'time_origin': args.start},
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.command == 'validate-natre':
        config.update(bounds=[-38, -20, 19, 32], start='2014-01-01', end='2024-01-01', cell_size=None)
        config['summary_config'].update(time_origin='2014-01-01')
        return 0 if validate_natre(config, args)['passed'] else 2
    if args.command == 'inventory':
        output = _data_directory(args.output)
        inventory = {}
        for _ in iter_local_index(config, inventory):
            pass
        _json_write(output / 'inventory.json', inventory)
        print(json.dumps(inventory, indent=2))
        return 0
    width = (east - west) % 360 or 360
    large_request = width > 30 or north - south > 30
    if large_request:
        if not args.validation_report:
            raise ValueError('Global/large runs require --validation-report from a passing NATRE validation')
        validation = json.loads(Path(args.validation_report).read_text())
        if not (validation.get('passed') and validation.get('global_authorized')
                and validation.get('script_sha256') == config['code_sha256']
                and validation.get('gdac') == config['gdac']
                and validation.get('source_snapshot') == _source_snapshot(config)):
            raise ValueError('NATRE validation did not pass for this script/archive; global processing is blocked')
    if config['cell_size'] is None and large_request:
        raise ValueError('Large/global requests require regional summary cells (--cell-size)')
    run_pipeline(config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
