import os
from datetime import datetime

import panel as pn
pn.extension("matplotlib")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import colors
import cmocean.cm as cmo

# -------------------------------------------------------------------
# Load datasets
# -------------------------------------------------------------------
ds_pbar = xr.open_dataset('/swot/SUM05/amf2288/ds_pbar_var.nc')
ds_pbar_grid = xr.open_dataset('/swot/SUM05/amf2288/ds_pbar_grid_var.nc')
ds_pbar_grad = xr.open_dataset('/swot/SUM05/amf2288/ds_pbar_grad_cip_masking.nc')

ds_pbar_grid = ds_pbar_grid.assign_coords(
    LON_left=ds_pbar_grad.LON_left,
    LAT_left=ds_pbar_grad.LAT_left
)
ds_pbar_grid = xr.merge([ds_pbar_grid, ds_pbar_grad.drop_vars({'DENSITY'})])

K_rho_p = xr.open_dataarray('/swot/SUM05/amf2288/K_rho_filt_p.nc').drop_vars('Z').rename({'PRESSURE': 'PRESSURE_mean'})
K_rho_p = K_rho_p.bfill(dim='PRESSURE_mean').ffill(dim='PRESSURE_mean')

# -------------------------------------------------------------------
# Normalize DataArray → Dataset
# -------------------------------------------------------------------
def ensure_dataset(obj, default_name):
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=obj.name or default_name)
    return obj

K_rho_p = ensure_dataset(K_rho_p, "K_rho")
ds_pbar_grid = ensure_dataset(ds_pbar_grid, "var")

# -------------------------------------------------------------------
# Variable list
# -------------------------------------------------------------------
vars_Krho = list(K_rho_p.data_vars.keys())
vars_grid = list(ds_pbar_grid.data_vars.keys())
all_vars = vars_Krho + vars_grid

# -------------------------------------------------------------------
# Colormap options
# -------------------------------------------------------------------
cmap_options = sorted([
   "cmo.balance", "cmo.dense", "cmo.dense_r", "cmo.gray", "cmo.gray_r",
   "cmo.haline", "cmo.matter_r", "cmo.thermal",
])

cmap_lookup = {
    "cmo.balance": cmo.balance,
    "cmo.dense": cmo.dense,
    "cmo.dense_r": cmo.dense_r,
    "cmo.gray": cmo.gray,
    "cmo.gray_r": cmo.gray_r,
    "cmo.haline": cmo.haline,
    "cmo.matter_r": cmo.matter_r,
    "cmo.thermal": cmo.thermal,
}

# -------------------------------------------------------------------
# Helper: get variable from correct dataset
# -------------------------------------------------------------------
def get_var(name):
    if name in vars_Krho:
        return K_rho_p[name]
    else:
        return ds_pbar_grid[name]

# -------------------------------------------------------------------
# MAP PLOTTING
# -------------------------------------------------------------------
def plot_map(ax, var, depth, cmap, scale, vmin, vmax):
    da = get_var(var).sel(PRESSURE_mean=depth, method="nearest")

    lon2d, lat2d = np.meshgrid(da["LON"], da["LAT"], indexing="ij")

    cmap_obj = cmap_lookup.get(cmap, cmap)

    vmin = float(vmin)
    vmax = float(vmax)

    norm = colors.LogNorm(vmin=vmin, vmax=vmax) if scale == "log" else colors.Normalize(vmin=vmin, vmax=vmax)

    pcm = ax.pcolormesh(
        lon2d, lat2d, da.values,
        cmap=cmap_obj,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, alpha=0.8)
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label(var)

    ax.set_title(f"{var} at {depth} m")

def make_map_figure(var, depth, cmap, scale, vmin, vmax):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-60))
    plot_map(ax, var, depth, cmap, scale, vmin, vmax)
    fig.tight_layout()
    return fig

# -------------------------------------------------------------------
# SECTION PLOTTING
# -------------------------------------------------------------------
def plot_section(ax, var, lon, cmap, scale, vmin, vmax):
    da = get_var(var)

    lon_sel = da["LON"].sel(LON=lon, method="nearest")
    section = da.sel(LON=lon_sel)

    lat = section["LAT"]
    depth = section["PRESSURE_mean"]

    LAT2D, DEPTH2D = np.meshgrid(lat, depth)

    cmap_obj = cmap_lookup.get(cmap, cmap)

    vmin = float(vmin)
    vmax = float(vmax)

    norm = colors.LogNorm(vmin=vmin, vmax=vmax) if scale == "log" else colors.Normalize(vmin=vmin, vmax=vmax)

    pcm = ax.pcolormesh(
        LAT2D, DEPTH2D, section.values,
        cmap=cmap_obj,
        norm=norm,
        shading="auto",
    )

    ax.invert_yaxis()
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{var} section at lon={float(lon_sel.values):.1f}")

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label(var)

def make_section_figure(var, lon, cmap, scale, vmin, vmax):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_section(ax, var, lon, cmap, scale, vmin, vmax)
    fig.tight_layout()
    return fig

# -------------------------------------------------------------------
# WIDGETS — MAP PANELS
# -------------------------------------------------------------------
var_m1 = pn.widgets.Select(name="Variable (Map 1)", options=all_vars, value=all_vars[0])
depth_m1 = pn.widgets.IntSlider(name="Depth (m)", start=0, end=2000, step=100, value=1000)
cmap_m1 = pn.widgets.Select(name="Colormap", options=cmap_options, value="cmo.dense_r")
scale_m1 = pn.widgets.Select(name="Scale", options=["linear", "log"], value="log")
vmin_m1 = pn.widgets.TextInput(name="vmin", value="5e-6")
vmax_m1 = pn.widgets.TextInput(name="vmax", value="1e-4")

var_m2 = pn.widgets.Select(name="Variable (Map 2)", options=all_vars, value=all_vars[0])
depth_m2 = pn.widgets.IntSlider(name="Depth (m)", start=0, end=2000, step=100, value=1000)
cmap_m2 = pn.widgets.Select(name="Colormap", options=cmap_options, value="cmo.dense_r")
scale_m2 = pn.widgets.Select(name="Scale", options=["linear", "log"], value="log")
vmin_m2 = pn.widgets.TextInput(name="vmin", value="5e-6")
vmax_m2 = pn.widgets.TextInput(name="vmax", value="1e-4")

# -------------------------------------------------------------------
# WIDGETS — SECTION PANELS
# -------------------------------------------------------------------
var_s1 = pn.widgets.Select(name="Variable (Section 1)", options=all_vars, value=all_vars[0])
lon_s1 = pn.widgets.FloatSlider(name="Longitude", start=-180, end=180, step=1, value=0)
cmap_s1 = pn.widgets.Select(name="Colormap", options=cmap_options, value="cmo.dense_r")
scale_s1 = pn.widgets.Select(name="Scale", options=["linear", "log"], value="log")
vmin_s1 = pn.widgets.TextInput(name="vmin", value="5e-6")
vmax_s1 = pn.widgets.TextInput(name="vmax", value="1e-4")

var_s2 = pn.widgets.Select(name="Variable (Section 2)", options=all_vars, value=all_vars[0])
lon_s2 = pn.widgets.FloatSlider(name="Longitude", start=-180, end=180, step=1, value=0)
cmap_s2 = pn.widgets.Select(name="Colormap", options=cmap_options, value="cmo.dense_r")
scale_s2 = pn.widgets.Select(name="Scale", options=["linear", "log"], value="log")
vmin_s2 = pn.widgets.TextInput(name="vmin", value="5e-6")
vmax_s2 = pn.widgets.TextInput(name="vmax", value="1e-4")

# -------------------------------------------------------------------
# REACTIVE FIGURES
# -------------------------------------------------------------------
fig_m1 = pn.bind(make_map_figure, var=var_m1, depth=depth_m1, cmap=cmap_m1, scale=scale_m1, vmin=vmin_m1, vmax=vmax_m1)
fig_m2 = pn.bind(make_map_figure, var=var_m2, depth=depth_m2, cmap=cmap_m2, scale=scale_m2, vmin=vmin_m2, vmax=vmax_m2)

fig_s1 = pn.bind(make_section_figure, var=var_s1, lon=lon_s1, cmap=cmap_s1, scale=scale_s1, vmin=vmin_s1, vmax=vmax_s1)
fig_s2 = pn.bind(make_section_figure, var=var_s2, lon=lon_s2, cmap=cmap_s2, scale=scale_s2, vmin=vmin_s2, vmax=vmax_s2)

pane_m1 = pn.pane.Matplotlib(fig_m1, tight=True)
pane_m2 = pn.pane.Matplotlib(fig_m2, tight=True)
pane_s1 = pn.pane.Matplotlib(fig_s1, tight=True)
pane_s2 = pn.pane.Matplotlib(fig_s2, tight=True)

# -------------------------------------------------------------------
# SAVE ALL FOUR PANELS
# -------------------------------------------------------------------
save_button = pn.widgets.Button(name="Save All 4 Panels", button_type="primary")
save_status = pn.pane.Markdown("")

def save_all(event):
    os.makedirs("figures", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fname = f"figures/fourpanel_{timestamp}.png"

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=-60)},
    )

    # Top row: maps
    plot_map(axes[0, 0], var_m1.value, depth_m1.value, cmap_m1.value, scale_m1.value, vmin_m1.value, vmax_m1.value)
    plot_map(axes[0, 1], var_m2.value, depth_m2.value, cmap_m2.value, scale_m2.value, vmin_m2.value, vmax_m2.value)

    # Bottom row: sections (no projection)
    axes[1, 0].remove()
    axes[1, 1].remove()
    ax_s1 = fig.add_subplot(2, 2, 3)
    ax_s2 = fig.add_subplot(2, 2, 4)

    plot_section(ax_s1, var_s1.value, lon_s1.value, cmap_s1.value, scale_s1.value, vmin_s1.value, vmax_s1.value)
    plot_section(ax_s2, var_s2.value, lon_s2.value, cmap_s2.value, scale_s2.value, vmin_s2.value, vmax_s2.value)

    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    save_status.object = f"Saved 4‑panel figure to `{fname}`"

save_button.on_click(save_all)

# -------------------------------------------------------------------
# LAYOUT
# -------------------------------------------------------------------
controls_m1 = pn.Column("### Map 1 Controls", var_m1, depth_m1, cmap_m1, scale_m1, vmin_m1, vmax_m1, width=260)
controls_m2 = pn.Column("### Map 2 Controls", var_m2, depth_m2, cmap_m2, scale_m2, vmin_m2, vmax_m2, width=260)

controls_s1 = pn.Column("### Section 1 Controls", var_s1, lon_s1, cmap_s1, scale_s1, vmin_s1, vmax_s1, width=260)
controls_s2 = pn.Column("### Section 2 Controls", var_s2, lon_s2, cmap_s2, scale_s2, vmin_s2, vmax_s2, width=260)

row_maps = pn.Row(controls_m1, pane_m1, pane_m2, controls_m2)
row_secs = pn.Row(controls_s1, pane_s1, pane_s2, controls_s2)

dashboard = pn.Column(
    "# Four‑Panel Dashboard",
    row_maps,
    row_secs,
    pn.Row(save_button, save_status),
)

dashboard.servable()
