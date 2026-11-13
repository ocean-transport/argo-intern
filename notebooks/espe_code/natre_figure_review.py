"""Read-only checks and rendering of saved NATRE summaries.

No downloads, profile processing, bootstrap draws, or file writes occur here.
Original plotting cells are executed without their active savefig statements
or standalone plt.show() calls; displaying the figure is managed by the caller.
Cells 6, 10 and 23 replace reductions with the corresponding saved pipeline
statistics; no bootstrap draws or missing arrays are fabricated.
A figure regenerated from a historical summary is a reference rendering, not
proof that a local GDAC processing run reproduced the historical analysis.
"""
from __future__ import annotations

import ast
import builtins
from datetime import datetime
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

SOURCE_NOTEBOOK = Path(__file__).with_name("plots_mixing_NATRE_gamman_analysis.ipynb")
SUMMARY_FIGURE_CELLS = (6, 9, 10, 17, 20, 21, 22, 23, 24)  # zero-based indices


def _missing_mask(a):
    if a.dtype.kind in "fc":
        return np.isnan(a)
    if a.dtype.kind in "mM":
        return np.isnat(a)
    return np.zeros(a.shape, dtype=bool)


def compare_datasets(reference, candidate):
    """Compare all variables positionally, including coordinates, without alignment.

    Exact values treat matching NaN/NaT positions as equal. Dtype and bitwise
    equality are reported separately. Dataset attributes are provenance rather
    than scientific arrays and are not included in this numerical comparison.
    Extra pipeline diagnostics prevent dataset identity but do not by themselves
    invalidate equality of all historical reference variables.
    """
    rows = []
    for name in sorted(set(reference.variables) | set(candidate.variables)):
        row = {"variable": name, "coordinate": name in reference.coords or name in candidate.coords}
        if name not in reference or name not in candidate:
            row.update(status="extra_candidate" if name not in reference else "missing_candidate",
                       values_exact=False, bitwise_equal=False)
            rows.append(row)
            continue
        a, b = reference[name], candidate[name]
        row.update(reference_dims=list(a.dims), candidate_dims=list(b.dims),
                   reference_shape=list(a.shape), candidate_shape=list(b.shape),
                   reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype),
                   dtype_equal=a.dtype == b.dtype)
        if a.dims != b.dims or a.shape != b.shape:
            row.update(status="dimension_mismatch", values_exact=False, bitwise_equal=False)
            rows.append(row)
            continue
        x, y = np.asarray(a.values), np.asarray(b.values)
        xm, ym = _missing_mask(x), _missing_mask(y)
        masks_equal = bool(np.array_equal(xm, ym))
        with np.errstate(invalid="ignore"):
            equal = np.equal(x, y) | (xm & ym)
        exact = bool(np.all(equal))
        bitwise = bool(x.dtype == y.dtype and x.dtype.kind != "O" and x.tobytes() == y.tobytes())
        row.update(status="equal" if exact else "value_mismatch", values_exact=exact,
                   bitwise_equal=bitwise, missing_mask_equal=masks_equal,
                   different_values=int(np.count_nonzero(~equal)), max_abs_difference=None)
        if x.dtype.kind in "biufc" and y.dtype.kind in "biufc":
            both = np.isfinite(x) & np.isfinite(y)
            if np.any(both):
                difference = np.abs(x[both].astype(np.clongdouble if x.dtype.kind == 'c' or y.dtype.kind == 'c' else np.longdouble)
                                    - y[both].astype(np.clongdouble if x.dtype.kind == 'c' or y.dtype.kind == 'c' else np.longdouble))
                row["max_abs_difference"] = float(np.max(difference))
            row["nonfinite_mismatches"] = int(np.count_nonzero((~np.isfinite(x) | ~np.isfinite(y)) & ~equal))
        elif x.dtype.kind in "mM" and y.dtype.kind == x.dtype.kind:
            both = ~xm & ~ym
            if np.any(both):
                unit = "datetime64[ns]" if x.dtype.kind == "M" else "timedelta64[ns]"
                delta = x[both].astype(unit).astype(np.longdouble) - y[both].astype(unit).astype(np.longdouble)
                row["max_abs_difference_ns"] = float(np.max(np.abs(delta)))
        rows.append(row)
    ref_rows = [r for r in rows if r["variable"] in reference.variables]
    dimensions_equal = dict(reference.sizes) == dict(candidate.sizes)
    reference_dimensions_equal = all(candidate.sizes.get(name) == size
                                     for name, size in reference.sizes.items())
    reference_exact = reference_dimensions_equal and all(r["values_exact"] for r in ref_rows)
    reference_bitwise = reference_dimensions_equal and all(r["bitwise_equal"] for r in ref_rows)
    return {"summary": {
        "dimensions_equal": dimensions_equal,
        "reference_dimensions_equal": reference_dimensions_equal,
        "reference_variables": len(reference.variables),
        "candidate_variables": len(candidate.variables),
        "missing_candidate_variables": sum(r["status"] == "missing_candidate" for r in rows),
        "extra_candidate_variables": sum(r["status"] == "extra_candidate" for r in rows),
        "reference_variables_exact": sum(r["values_exact"] for r in ref_rows),
        "all_reference_variables_exact": reference_exact,
        "reference_variables_bitwise": sum(r["bitwise_equal"] for r in ref_rows),
        "all_reference_variables_bitwise": reference_bitwise,
        "dataset_arrays_bitwise": reference_bitwise and dimensions_equal and len(reference.variables) == len(candidate.variables),
        "coordinates_bitwise": all(r["bitwise_equal"] for r in ref_rows if r["coordinate"]),
        "dataset_values_exact": reference_exact and dimensions_equal and len(reference.variables) == len(candidate.variables),
        "coordinates_exact": all(r["values_exact"] for r in ref_rows if r["coordinate"]),
        "attributes_compared": False,
        "alignment_or_tolerances_used": False,
    }, "variables": rows}


def summary_namespace(ds):
    """Map saved summary names to the original plotting notebook's globals."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    namespace = {"np": np, "plt": plt, "cm": cm, "colors": colors, "datetime": datetime}
    for name, value in ds.variables.items():
        if set(value.dims) == {"SG", "TIME"}:
            value = value.transpose("SG", "TIME")
        array = np.array(value.values, copy=True)
        namespace[name] = array
        namespace[name + "_sg"] = array
    aliases = {"date": "DATES", "PSdia": "PdiaS_sg", "PSdia_lo": "PdiaS_sg_lo",
               "PSdia_hi": "PdiaS_sg_hi", "Piso_profile_lo": "Piso_sg_lo",
               "Piso_profile_hi": "Piso_sg_hi", "PSiso_profile_lo": "PSiso_sg_lo",
               "PSiso_profile_hi": "PSiso_sg_hi", "profile_time": "time",
               "finescale_mean_resolution": "mean_res", "epsilon_finescale_cleaned": "epsilon0",
               "iso_slope_x": "sx_sg", "iso_slope_y": "sy_sg",
               "iso_slope_x_pvalue": "sx_sg_p", "iso_slope_y_pvalue": "sy_sg_p",
               "iso_slope_x_manual_pvalue": "sx_pval"}
    for source, target in aliases.items():
        if source in namespace:
            namespace[target] = namespace[source]
    if "Z" in ds.coords:
        namespace["z"] = np.array(ds.Z.values, copy=True)
    if "Zfs" in ds.coords:
        namespace["Z"] = np.array(ds.Zfs.values, copy=True)
    for source, target in {"Krho": "Krho", "chi": "chi", "chiS": "chiS", "Pdia": "Pdia", "PSdia": "PdiaS"}.items():
        name = source + "_bootstrap"
        if name in ds:
            namespace[target + "_sg_boot_all"] = np.array(ds[name].transpose("SG", "TIME", "bootstrap").values, copy=True)
    # Cell 23 inherits these means from cell 22; compute the identical display
    # reduction if raw bootstrap arrays are explicitly available in a dataset.
    for name in ("Piso", "PSiso"):
        if name in namespace:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                namespace[name + "_mean"] = np.nanmean(namespace[name], axis=1)
    return namespace


def _plot_source(cell_index, source_notebook=None):
    if cell_index not in SUMMARY_FIGURE_CELLS:
        raise ValueError(f"Supported original summary figure cells: {SUMMARY_FIGURE_CELLS}")
    path = Path(source_notebook or SOURCE_NOTEBOOK)
    source = "".join(json.loads(path.read_text())["cells"][cell_index]["source"])
    if cell_index == 6:
        # The original loop's integer-truncated percentages are already saved.
        source = ("mr = np.r_[resolution_bin_lower, resolution_bin_upper[-1]]\n"
                  "chi_count = resolution_missing_chi_percent\n" + source[source.index("fig, ax ="):])
    elif cell_index == 10:
        source = source.replace("np.nanmean(epsilon/Krho*GAMMA_mix, axis = 1)",
                                "epsilon_Krho_stratification_proxy_depth_mean")
        source = source.replace("np.nanmean(epsilon, axis = 1)", "epsilon_depth_mean")
    elif cell_index == 23:
        # Only replace the 15 bootstrap reductions. Plot styling, residual
        # means inherited from cell 22, and pooled residual bounds stay intact.
        replacements = {"Krho": "Krho", "chi": "chi", "chiS": "chiS",
                        "Pdia": "Pdia", "PdiaS": "PSdia"}
        parsed = ast.parse(source)
        lines = source.splitlines(keepends=True)
        for node in parsed.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            for original, saved in replacements.items():
                for suffix, product in [("mean", "mean"), ("lo", "p00_5"), ("hi", "p97_5")]:
                    if target == original + "_" + suffix:
                        lines[node.lineno - 1] = f"{target} = {saved}_pooled_bootstrap_{product}\n"
                        for j in range(node.lineno, node.end_lineno):
                            lines[j] = "\n"
        source = "".join(lines)
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    # Keep all plotting formulas, labels, limits and ordering exactly as written.
    # Remove standalone savefig calls and caller-managed plt.show() display.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and (node.value.func.attr == "savefig" or
                     (node.value.func.attr == "show" and isinstance(node.value.func.value, ast.Name)
                      and node.value.func.value.id == "plt"))):
            for i in range(node.lineno - 1, node.end_lineno):
                lines[i] = "\n"
    return "".join(lines), path


def figure_requirements(ds, cell_index, source_notebook=None):
    """List original notebook globals unavailable from this saved dataset."""
    source, _ = _plot_source(cell_index, source_notebook)
    tree = ast.parse(source)
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    stores = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    imports = {a.asname or a.name.split(".")[0] for n in ast.walk(tree)
               if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    return sorted(loads - stores - imports - set(summary_namespace(ds)) - set(vars(builtins)))


def render_summary_figure(ds, cell_index, source_notebook=None):
    """Render one original summary figure; never recompute the analysis.

    Cells 6, 10 and 23 use saved numerical plot products for the original
    reductions. A historical summary without pooled statistics cannot recreate
    cell 23's intervals here. No bootstrap sample is regenerated.
    Rendering is not a pixel-equality claim: Matplotlib/fonts and original saved
    notebook execution state also affect output.
    """
    source, path = _plot_source(cell_index, source_notebook)
    missing = figure_requirements(ds, cell_index, path)
    if missing:
        raise ValueError(f"Original cell {cell_index} cannot be rendered from this saved summary; missing: {', '.join(missing)}")
    namespace = summary_namespace(ds)
    plt = namespace["plt"]
    before = set(plt.get_fignums())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            exec(compile(source, f"{path}:cell-{cell_index}", "exec"), namespace)
        figure = namespace["fig"]
        figure.natre_review_source = {"notebook": str(path), "cell_index": cell_index,
                                     "plot_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                     "savefig_disabled": True, "display_managed_by_caller": True,
                                     "saved_statistic_adapter": cell_index in (6, 10, 23)}
        return figure
    except Exception:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
        raise


def render_profile_figure(profiles, source_notebook=None):
    """Render original cell 11 from saved full-profile CT/SA/Z arrays.

    The caller must supply the intended profile order. This function performs
    no sorting, profile processing, or file writes; the original every-50th
    profile selection and sigma0 contour calculation remain unchanged.
    """
    import matplotlib.pyplot as plt
    import gsw
    required = {"CT", "SA", "Z"}
    missing = required - set(profiles.variables)
    if missing:
        raise ValueError("Profile T-S figure requires saved arrays: " + ", ".join(sorted(missing)))
    for name in ("CT", "SA"):
        if set(profiles[name].dims) != {"Z", "time"}:
            raise ValueError(f"{name} requires dimensions Z and time; got {profiles[name].dims}")
    if profiles.Z.dims != ("Z",):
        raise ValueError("Z must be the one-dimensional depth coordinate")
    path = Path(source_notebook or SOURCE_NOTEBOOK)
    source = "".join(json.loads(path.read_text())["cells"][11]["source"])
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and (node.value.func.attr == "savefig" or
                     (node.value.func.attr == "show" and isinstance(node.value.func.value, ast.Name)
                      and node.value.func.value.id == "plt"))):
            for i in range(node.lineno - 1, node.end_lineno):
                lines[i] = "\n"
    source = "".join(lines)
    namespace = {"np": np, "plt": plt, "gsw": gsw,
                 "CT": np.array(profiles.CT.transpose("Z", "time").values, copy=True),
                 "SA": np.array(profiles.SA.transpose("Z", "time").values, copy=True),
                 "z": np.array(profiles.Z.values, copy=True)}
    before = set(plt.get_fignums())
    try:
        exec(compile(source, f"{path}:cell-11", "exec"), namespace)
        figure = namespace["fig"]
        figure.natre_review_source = {"notebook": str(path), "cell_index": 11,
                                     "plot_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                     "savefig_disabled": True, "display_managed_by_caller": True,
                                     "profile_order_changed": False,
                                     "input_profile_count": int(profiles.sizes["time"])}
        return figure
    except Exception:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
        raise
