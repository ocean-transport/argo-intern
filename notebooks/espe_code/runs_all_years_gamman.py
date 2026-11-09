import argopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.interpolate as intrp
import gsw
import cmocean
import matplotlib.cm as cm
import sys
import os
import warnings
import seawater as sw
import mixsea as mx
import neutral_density as nds #using this package instead of local directory
from tenacity import retry, stop_after_attempt, wait_fixed

warnings.filterwarnings("ignore")
dz = 2. #defines the vertical grid for initial window averaging
min_resolution_micro = 10. #resolution used to discard fine scale segments
min_mean_res_in_upper_1000m = 5. #discard profiles that have lower resolution in the upper 1000 m


def mixed_layer_depth(z0, den0, Dd = 0.03, crit = "diff", z_min = 30., intrp = True):
    #Mixed layer calculation
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


YR = np.arange(2021,2024,1)
for yr in YR:
    print("\n\nRunning %d"%(yr))
    
    print(yr)

    #define region. In this example, a box around NATRE
    with argopy.set_options(mode='research'):
        ds0 = argopy.DataFetcher().region([-32-6, -26+6, 23-4, 28+4, 0, 2200, '%04d-01'%(yr), '%04d-01'%(yr+1)]).load()

    #converts to profile
    DS = ds0.data.argo.point2profile()

    #READS and saves to netCDF
    output_dir = "/swot/SUM05/amf2288/espe_code/data_files"
    os.makedirs(output_dir, exist_ok=True)

    filename_raw = os.path.join(output_dir, "NATRE_%04d_%04d_Argo_resmode.nc" % (yr, yr+1))
    DS.to_netcdf(filename_raw)

    Z = np.arange(2,2000+dz,dz)
    
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

    print("Year %d, Number of profiles: %d, Number of good profiles: %d"%(yr,N_PROF0, N_PROF))
    Time = np.array(DS.TIME.compute())[iigood]

    Lon = np.array(DS.LONGITUDE.compute())[iigood]
    Lat = np.array(DS.LATITUDE.compute())[iigood]

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

        p0= np.array(DS.isel(N_PROF=j).PRES.compute())
        t0 = np.array(DS.isel(N_PROF=j).TEMP.compute())
        s0 = np.array(DS.isel(N_PROF=j).PSAL.compute())
        ii = np.isfinite(t0+s0)

        t0 = t0[ii]
        s0 = s0[ii]
        p0 = p0[ii]

        sa0 = gsw.SA_from_SP(s0, p0, Lon[i], Lat[i]);
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
    print("Gamman calculation")
    gamman = np.full((NP, N_PROF), np.nan)
    for j in range(N_PROF):
        ii = np.isfinite(PSAL[:,j])
        if np.sum(ii) == 0:
            continue
        gamman[ii,j], _, _ = nds.gamma_n(PSAL[ii,j],TEMP[ii,j],P[ii,j],Lon[j],Lat[j])


    gamman = np.array(gamman)
    N2 = np.full(gamman.shape, np.nan)
    for i in range(N_PROF):

        gamman0 = np.copy(gamman[:,i])
        ii = np.isfinite(gamman0)
        z00 = Zm[ii,i]
        gamman00 = gamman0[ii]
        gamman00 = np.sort(gamman00)
        N20 = np.full(gamman00.size, np.nan)
        N20[1:-1] = 9.81/(1000+gamman00[1:-1])*(gamman00[2:]-gamman00[:-2])/(z00[2:]-z00[:-2])

        N20[0] = N20[1]
        N20[-1] = N20[-2]
        N2[ii,i] = N20
    
    
    ##mixing
    print("Finescale") #from https://mixsea.readthedocs.io/en/v0.1.2/
    # Center points of depth windows. Windows are half overlapping, i.e.
    # their size (200m) is double the spacing here (100m).
    window_size = 200
    min_size = 10.
    dZ = window_size // 4
    #print("window size {} m, window spacing {} m".format(window_size, dz))
    depth_bin = np.arange(dZ, 2000, dZ)
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
        if np.sum(np.isfinite(TEMP[:,i]+PSAL[:,i]))<100:
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
            print(f"Skipped profile at index {i} due to error: {e}")
            continue
            
        z0 = diag['depth_bin']
        for j in range(z0.size):
            jj = np.where(depth_bin == z0[j])[0]
            if jj.size == 0:
                continue
            eps_st[jj[0],i] =  diag['eps_st'][j]
            Krho_st[jj[0],i] = diag['krho_st'][j]
            
              
    #polynom
    print("Smooth fields")
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
            pt = np.polyfit(gamman[jj,i],CT[jj,i],POLY_ORDER)
            ppt = np.poly1d(pt)
            CT_mean[j,i] = ppt(sg1s[j])#ppt(SG1[j,i])


            ps = np.polyfit(gamman[jj,i],SA[jj,i],POLY_ORDER)
            pps = np.poly1d(ps)
            SA_mean[j,i] = pps(sg1s[j])#pps(SG1[j,i])
            
            
            
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

    dZ = depth_bin[1]-depth_bin[0]
    ZZ = np.r_[depth_bin-dZ/2, depth_bin[-1]+dZ/2]

    fig, ax = plt.subplots(1,2, sharey = True, figsize = (8,5))

    ax[0].stairs(np.nanmean(chi, axis = 1), ZZ, orientation = "horizontal", color = "k", lw = 2)
    ax[0].stairs(np.nanmean(chi_e, axis = 1), ZZ, orientation = "horizontal", color = "k", lw = 2, ls = "dotted")
    ax[0].stairs(np.nanmean(Pdia, axis = 1), ZZ, orientation = "horizontal")
    ax[0].stairs(np.nanmean(Pdia_e, axis = 1), ZZ, orientation = "horizontal", color = cm.tab10(0), ls = "dotted")
    ax[0].stairs(np.nanmean(chi-Pdia, axis = 1), ZZ, orientation = "horizontal")
    ax[0].stairs(np.nanmean(chi_e-Pdia_e, axis = 1), ZZ, orientation = "horizontal", color = cm.tab10(1), ls = "dotted")
    ax[0].set_xscale("log")
    ax[0].set_xlim((1e-11, 1e-7))
    ax[0].set_ylim((2000,0))
    ax[0].grid(True)
    ax[0].set_xlabel("$\\chi_{\\Theta}$")
    ax[0].set_ylabel("Depth (m)")
    ax[1].stairs(np.nanmean(chiS, axis = 1), ZZ, orientation = "horizontal", color = "k", lw = 2)
    ax[1].stairs(np.nanmean(PdiaS, axis = 1), ZZ, orientation = "horizontal")
    ax[1].stairs(np.nanmean(chiS-PdiaS, axis = 1), ZZ, orientation = "horizontal")
    ax[1].set_xscale("log")
    ax[1].set_xlim((1e-13, 1e-8))
    ax[1].set_ylim((2000,0))
    ax[1].grid(True)
    ax[1].set_xlabel("$\\chi_S$")
    fig.tight_layout()

    #fig.savefig("figures_NATRE_%dm/mean_chi_profiles_%d_NATRE_gamman.png"%(dz,yr))



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
            attrs = {"Description":"Fine-scale mixing estimates at NATRE from Argo floats"},
        )
    ds.Z.attrs['Description'] = 'Depth coordinate for fully resolved data (m)'
    ds.Zfs.attrs['Description'] = 'Depth coordinate for fine scale parameters (m)'
    ds.time.attrs['Description'] = 'Time coordinate'
    print(ds)

    filename = os.path.join(output_dir, "Mixing_Argo_NATRE_%dm_%04d_resmode.nc" % (dz, yr))
    ds.to_netcdf(filename)
    
DS = xr.open_mfdataset(os.path.join(output_dir, "Mixing_Argo_NATRE_%dm_*_resmode.nc" % dz),
                       concat_dim='time', combine='nested')
DS.to_netcdf(os.path.join(output_dir, "Mixing_Argo_NATRE_%dm_resmode_ALL.nc" % dz))
