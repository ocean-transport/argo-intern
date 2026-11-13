**Argo mixing workflow: handoff for the next session**

**November 13 completion — refreshed local NATRE replication PASSED. Read first.**

The user reaffirmed the purpose: NATRE is the regression case for a general
local-archive processing pipeline that can subsequently process global Argo.
Reference values must not be inserted into local outputs. Global execution is
still a separate step after reviewing NATRE.

The remaining input discrepancy was resolved by inspecting the **float aggregate
`<WMO>_prof.nc` files already present in the same frozen November 9 snapshot**.
They differ from the mono-cycle files in that snapshot. Three controls spanning
salinity and pressure/temperature differences reproduced the historical arrays
directly from aggregate files. A complete independent diagnostic then read all
193 aggregate files covering the 13,551 NATRE index candidates:

- All **10,483** saved in-range raw profiles have **bitwise-identical pressure,
  temperature, and salinity** after primary selection and research QC.
- Standard `netCDF4.num2date` CF decoding, at its microsecond resolution,
  reproduces **all 10,483 reference timestamps exactly**. No rounding to whole
  seconds is applied. This avoids the old xarray nanosecond decoding artifacts;
  a fractional-second regression test ensures genuine fractional times survive.
- The accepted physical population is exactly **6,261**, with zero missing or
  extra identities. All accepted profiles have exact P/T/S, coordinates, and CF
  time in this diagnostic. No source files are missing or unreadable.
- This establishes input parity for NATRE from the refreshed local archive,
  without empirical P/T/S rounding or a reference-value lookup. It does not
  by itself establish the subsequent scientific-output comparison.

The complete report is
`/swot/SUM05/amf2288/argo_mixing/natre_replication_20261113/aggregate_input_audit.json`.
`verify_argo_snapshot.py --source-layout multi-only` reproduces this diagnostic;
the default `mono-preferred` mode preserves the earlier source-selection policy.
The alternate CF timestamp comparison is recorded alongside ordinary xarray
decoding. Earlier claims that local input differences remain unresolved describe
the **mono-cycle** investigation and are superseded by this aggregate result.

Production changes in `process_argo_mixing.py`:

- The default source is the refreshed archive and `--local-layout aggregate`.
  The core index still selects physical profiles spatially and temporally;
  their data come from the corresponding local aggregate file. There is no
  remote fetch and no fallback to historical arrays. `--local-layout mono`
  remains an explicit diagnostic choice.
- Indexed identities are resolved before selecting primary sampling. Multiple
  primary entries and absent indexed identities raise errors. Secondary-only
  samples are excluded explicitly. The selected profiles then use the installed
  Argopy research QC, retaining unrounded pressure and inclusive 2,200 dbar bounds.
- Batches group reads by aggregate file; only requested profiles and needed
  variables are loaded. Dates use standard CF decoding. Global profile order
  is chronological with the signed Argopy physical UID as the tie breaker,
  independently of references, batch sizes, and worker completion order. That
  ordering matches the complete historical NATRE order, including all six ties.
- Comparison acceptance now requires equal dtype and array bytes, including
  NaN payloads and signed zeros. Profile coverage and order must match. Numerical
  tolerances are diagnostic only. The review helper exposes top-level bitwise
  results separately from value equality and complete schema identity.
- Successful validation computes the regional summary from **local partitions**,
  compares all historical deterministic variables bitwise, and reports historical
  bootstrap differences as exempt. The global gate checks the validated source
  layout and comparison policy as well as code and archive provenance.

The actual production loader passed a 24-profile integration control spanning
the accepted population: every P/T/S array, coordinate, and timestamp matched
the saved raw reference bitwise. All 15 regression tests in
`test_argo_replication.py` passed. AST comparison against the committed version
confirmed the profile, regional, bootstrap, and saved-figure scientific kernels
are unchanged; the changes concern loading, ordering, validation, and orchestration.

The full fresh validation **completed successfully**, using 12 workers,
64-profile batches, and 8 GiB per-worker budgets, at
`/swot/SUM05/amf2288/argo_mixing/natre_replication_20261113/validation`.
Its log is the adjacent `run.log`. The report `validation.json` records
`status: passed`, `passed: true`, and the passing global-eligibility gate for this
code/archive/layout. **No global processing was launched.** Review NATRE and
global aggregation settings before requesting global execution.

- The fresh reference replay completed all **169 batches**, with 6,261 accepted
  profiles and zero failed batches. All 34 historical variables and three
  coordinates pass bitwise comparison in the original order.
- The refreshed **local-source** stage completed all **212 batches**, with
  exactly 6,261 accepted profiles and zero failed batches. All 34 historical
  variables and three coordinates pass bitwise comparison, in the same order,
  with zero missing or extra profiles. Its 123 nonempty partitions are under
  `validation/local/profiles/region`.
- The full regional summary was calculated **from those local partitions**,
  with 1,000 bootstrap replicates and seed `20261110`. All **17 deterministic
  historical variables and both coordinates match bitwise**. The 32 historical
  bootstrap-dependent variables differ, as explicitly permitted. The complete
  local summary is `validation/local/summaries/region.nc`; it is not a subset
  or a reference-input result.

The producing script SHA256 is
`abc425c7f264d7eac5982bac99714b42dc27edeede1865fb7a29abbb3f2bde24`.
The original notebook and historical scientific datasets remain preserved.

**Direct replacement for the original notebook's profile dataset:**
`validation/local/NATRE_profiles_replica.nc`. This standalone export contains
only the historical 34 data variables and three coordinates, with identical
schema and descriptive attributes. **Every array is sourced from the new local
partitions.** `export_natre_replica.py` copies only the reference's variable
selection and descriptive metadata, then verifies all values against the
reference. It checks again after writing and reopening the NetCDF, using both
bitwise comparison and `xr.testing.assert_identical`. The adjacent
`NATRE_profiles_replica.json` records this pass, both file hashes, and source
provenance. The NetCDF container bytes differ because of encoding/compression;
the decoded array bytes, dtypes, schema, and descriptive attributes match.
Extra scientific diagnostics and physical identity fields remain in the original
local partitions and their manifests, rather than being discarded.

**Figures and review:** `NATRE_figure_replication_review.ipynb` now selects the
new local run, verifies its provenance and bitwise gate, independently compares
the saved deterministic summary arrays, and displays all ten historical figures
followed by all ten local-source renderings. It executed without errors and
contains 20 saved image outputs. Its scientific kernel was `Argo_Feb_25`; the
installed base-environment notebook controller supplied `nbclient`/`nbformat`.
No packages were installed or changed. The first controller attempt is recorded
in `figure_review_execution.log` (missing `nbformat`); the successful execution
is in `figure_review_execution_v2.log`. This did not affect scientific processing.
Standalone local PNGs are under `natre_replication_20261113/figures`, with names
`local_cell_06.png`, `local_cell_09.png`, etc., using original notebook cell
indices. `figure_review.json` records all ten paths and hashes. Original plotting
formulas and styles are retained; historical bootstrap realizations and identical
PNG pixels are not required. No reference arrays were substituted into local
figures. The review notebook remains read-only when executed.

`natre_replication_20261113/environment.json` records dependency versions, import
paths, and code/reference hashes. `test_argo_replication.py` contains the 15
passing regression tests. Dataset exports, reports, logs, figures, and Dask spill
are on `/swot`; only repository code/docs/notebooks changed in the workspace.
All processing, export, and notebook-execution jobs have finished. No scientific
or plotting work remains running from this validation.

**November 13 review — bitwise requirement clarified; saved outputs checked read-only.**

The user reiterated that the target is the figures in
`plots_mixing_NATRE_gamman_analysis.ipynb`, produced by the Python workflow using
the refreshed local Argo snapshot. **The underlying datasets must replicate the
historical reference bitwise. Bootstrap estimates need not reproduce the
historical random realization.** This supersedes any suggestion below to accept
numerical tolerances or choose a different reference population. Neither close
values, matching profile counts, nor visually similar figures establish success.
Bootstrap-dependent fields must be identified explicitly; their exemption does
not extend to deterministic inputs, profile calculations, or coordinates.

The handoff, saved validation report, refreshed input audit, production comparison
functions, and original/review notebook sources were reviewed. A new read-only
check used `natre_figure_review.compare_datasets`, inspecting each variable's
`bitwise_equal` result (equal dtype and C-order array bytes, with identical
dimensions/shapes). No alignment, sorting, dtype conversion, or NaN/signed-zero
normalization was applied:

- The 169 saved `legacy_raw/profiles/region/*.nc` partitions in
  `natre_validation_20261110`, concatenated in filename order, match **all 34
  historical profile variables and all three coordinates bitwise across all
  6,261 profiles**, directly against
  `espe_code/data_files/Mixing_Argo_NATRE_2m_resmode_ALL.nc`. Profile order already
  matches. This is confirmation of the saved reference-input replay, not a run
  from the refreshed local archive.
- The saved `reference_deterministic_check.nc` matches **all 17 deterministic
  historical summary variables and both coordinates bitwise** against the
  corresponding variables in `espe_code/NATRE_Argo_mixing_gamman.nc`.
- The replay contains **14 additional diagnostic/identity variables**. Thus
  historical-array parity is established, but complete dataset schema identity
  is not. File-container bytes and attributes were not compared. Do not describe
  these partition files as byte-identical copies of the historical NetCDF file.

Recounting the saved refreshed input audit for the **6,261 accepted profiles**
confirmed 3,933 bitwise pressure matches, 6,116 temperature matches, 1,749 salinity
matches, and 1,748 profiles with all three arrays bitwise equal. Coordinates match
numerically for all profiles; timestamps match exactly for 1,240. Only **1,139**
profiles satisfy the audit's combined array/coordinate/timestamp exactness check.
The local-source input mismatch is therefore still unresolved. There is still no
refreshed local scientific summary to validate or use for replicated figures.

**Validation implementation gaps identified:** production `_comparison_array`
uses `np.isclose`; profile and summary comparison defaults retain `rtol=1e-10`,
including validation calls that specify only `atol=0`. Even both tolerances set
to zero would not establish dtype, NaN-payload, or signed-zero byte equality.
The review helper reports per-variable byte equality, but its top-level
`all_reference_variables_exact` flag checks values, not bytes. Future acceptance
must explicitly require bitwise comparison and full population/order coverage,
with historical bootstrap estimates reported separately. Additional variables,
attributes, and serialization must also be reported separately from reference
array parity rather than silently treating different schemas as identical.

Next implementation work remains: repair primary-sampling selection; establish
the causes and exact conversion rules, if any, for the remaining local-versus-
ERDDAP differences; enforce the bitwise acceptance criterion; then produce and
validate a fresh NATRE result. Do not substitute saved reference values into a
purported local-source result or redefine the historical baseline to force a
pass. Global processing remains unauthorized.

This review changed only this handoff. No production/loader changes, scientific
rerun, bootstrap rerun, or data-file writes were made. The read-only checks
completed; their findings are recorded here and their detailed output was
printed in the session rather than saved as a new report.

**November 11 figure-review preparation — no new scientific run.**

The user is leaving shortly and requested a notebook for reviewing exact figure
replication, with global expansion only after discussion. A read-only review
notebook is now `notebooks/espe_code/NATRE_figure_replication_review.ipynb`, with
helper `natre_figure_review.py`. It displays all ten saved original figures as
explicitly labeled historical references. These are not outputs from the
refreshed local archive. `LOCAL_STAGE = None` is intentional: no refreshed local
mixing summary exists yet. The old 4,980-profile run is not selected as a
replacement. Future candidates must explicitly identify the refreshed archive,
NATRE selection, and 6,261 profiles, then undergo strict saved-array comparison.

The figure adapters read saved summaries/profile partitions. Cells 6, 10, and 23
use saved plot statistics; other drawing code retains the original formulas,
labels, and limits. Historical images do not establish that the current source
cells generated them; their provenance is the saved notebook. Numerical equality
and pixel equality are distinct. The original unseeded bootstrap realization
cannot be independently regenerated exactly without its RNG state/draws.

The review notebook's default cells were executed read-only and display the
historical images immediately. No Argo ingestion, mixing calculation, bootstrap
rerun, or global processing was launched. Next discussion should settle exact
input parity and bootstrap comparison before a fresh NATRE run. The original
plotting notebook and production script were preserved.

**November 11 snapshot refresh — complete. Read this section first.**

The user explicitly authorized updating the local snapshot to match the period
of Espe's November 2026 inputs. The saved raw attributes actually span
**November 8–9, 2026** (2014–2020 files on November 8; 2021–2023 on November 9).
Those are ERDDAP retrieval dates, not an immutable full-GDAC snapshot identifier.

A fixed official global snapshot dated **November 9, 2026** is available:
[DOI 10.17882/42182#138320](https://doi.org/10.17882/42182#138320).
Its archive is 97,398,616,816 bytes (90.71 GiB). The published checksum is
`029440622324ed0b003d851bf98cf6a2e0edc7e343a27af681c426c5d8b2cdc3`;
the downloader requires the complete file's SHA256 to match before finalizing.
Official catalogue/HTTP evidence is saved under
`/swot/SUM05/amf2288/argo_mixing/snapshot_refresh_20261111/metadata`.

The download completed with four bounded connections, and its full SHA256
**matches the published checksum exactly**. Its destination is
`/swot/SUM05/amf2288/argo_mixing/snapshot_refresh_20261111/download/138320.tar.gz`.
Read the adjacent `.download.json` for status and `completed_bytes`, and
`.download.log` for progress. The `.partial` file is pre-sized to the full archive
length; **its apparent size is not completed download progress**. The final
archive appears only after verification, accompanied by `.verified.json`.

Core-GDAC extraction and global index verification **completed successfully**
on November 11 at 21:39 UTC. All **3,434,826 indexed core profiles** exist as
regular files, with **zero missing paths**. The 11 DAC archives contain
**3,523,000 extracted files** (444,872,052,922 logical bytes, including aggregate
and supporting files). The completed extraction log is
`/swot/SUM05/amf2288/argo_mixing/snapshot_refresh_20261111/extraction_processes.log`.
The initial `extraction.log` ends with an intentional graceful interruption to
switch from threads to spawned process workers. Completed atomic files and DAC
markers are reused; input/output paths and validation rules are unchanged. The
process implementation passed synthetic resume, cancellation, and safety checks
and approximately doubled throughput in a bounded benchmark. Its SHA256 is
`ece25dad4575750b71aa833609378719f139ffe264cc9c0fe30cd7ad2d064082`.
The verified core-GDAC destination for subsequent work is
`/swot/SUM05/amf2288/argo_snapshots/gdac_20261109_138320`.
Its root `snapshot_manifest.json` and
`.snapshot_metadata/core_index_verification.json` both report `status: complete`.
The manifest retains `failed_utc`/`error` fields from the intentional earlier
interruption; the final status and verification report establish completion.
The old `/swot/SUM05/dbalwada/Argo_sync` is preserved. Core files and matching
indexes are expanded; BGC and geographic products remain available inside the
complete downloaded archive. The source index SHA256 is
`3e890546b97d11b6a930193f782d8913fc4892004e1461e3e36af77657bf4422`.

**Use the new path explicitly with `--gdac`.** The unchanged production script's
default still points to the old archive. No full mixing rerun or global
scientific processing was started. All download, extraction, and input-audit
jobs have finished; nothing remains running from this refresh.

Resume an interrupted download with the identical command; already completed
chunks are rehashed and reused:

```bash
python3 -B notebooks/espe_code/fetch_argo_snapshot.py \
  --url https://www.seanoe.org/data/00311/42182/data/138320.tar.gz \
  --bytes 97398616816 \
  --sha256 029440622324ed0b003d851bf98cf6a2e0edc7e343a27af681c426c5d8b2cdc3 \
  --output /swot/SUM05/amf2288/argo_mixing/snapshot_refresh_20261111/download/138320.tar.gz \
  --workers 4 --chunk-mib 256
```

The completed primary-input diagnostic is
`/swot/SUM05/amf2288/argo_mixing/snapshot_refresh_20261111/natre_refreshed_input_audit.json`.
It inspected all **13,551 NATRE candidate files**, with no missing files,
read errors, duplicate identities, or file/index update-date mismatches.
Using primary sampling, direct research QC, and the pressure-spacing criterion,
it accepts **exactly the same 6,261 physical identities as the historical
reference**, with **zero missing or extra profiles**. All 4,980 previously shared
profiles remain; every earlier deficit is recovered: 820 QC rejections,
258 sampling/spacing failures, 202 absent-index profiles, and one missing-index-
position profile. The 820 now pass the diagnostic QC; this supports archive age
and revision as the cause of their old rejection, not a conclusion that Espe
used bad data.

This is a **diagnostic with explicit primary-profile selection**, not the
unchanged production loader. Updating the source alone does not repair that
loader's merging of primary/secondary samples. Its sampling correction remains
necessary before a new scientific validation.

**Population parity is recovered; exact input-value parity is not.** Among the
6,261 historical accepted profiles, all retained-level counts and coordinates
match. Full arrays are bitwise equal for 3,933 pressure profiles, 6,116 temperature
profiles, and 1,749 salinity profiles; all three arrays match for 1,748 profiles.
Exact timestamps match for 1,240 profiles; all remaining offsets are at most
2.553344 ms, and the old two-day offsets are gone. Only 1,139 profiles match all
arrays, coordinates, and timestamps exactly. Largest absolute differences are
0.00048828125 dbar, 0.0005002021789550781 degrees C, and
0.000049591064453125 in practical salinity. Their complete causes remain to be
resolved; do not automatically attribute every difference to rounding or relax
the comparison.

The report's top-level precision totals use **10,483 saved raw profiles**, a
different denominator from the 6,261 accepted profiles above. All 10,483 raw
physical identities and their retained-level counts also match exactly.
Of all candidates, 1,571 fail delayed-mode selection, 67 fail position/time QC,
and one lies outside the selection according to its actual metadata. The
combined `spacing_or_insufficient_levels` category contains 1,429 profiles with
zero retained QC levels, 4,220 with mean spacing above 5 dbar, and two without
usable upper-1,000-dbar spacing pairs; do not describe all 5,651 as resolution
failures.

The audit does not calculate mixing metrics or pass the scientific validation
gate. Its `global_authorized` and `scientific_validation_gate_evaluated` fields
remain false. The production script is unchanged, SHA256
`7b3edfe77dcc7af6d085e20c1d92c07785d35f62abb9f21035e01136b88bd3e8`.
Next, correct sampling selection and resolve the remaining input representation
or revision differences, then validate in a **new output directory** against
the preserved reference. The official November 9 GDAC snapshot matches the
retrieval period; it is not claimed to be byte-identical to Espe's saved ERDDAP
responses.

The earlier index-only audit recorded that the refreshed index header is dated
`20261108172416` (November 8, 2026, 17:24:16 UTC). It contains **3,434,826 global
core rows** and **13,551 NATRE candidates** for 2014–2023, extending through
December 31, 2023. **All 6,261 historical accepted physical identities appear
both in the full index and in the requested NATRE selection**, with no missing
or duplicate expected identities. The completed extraction and primary-input
checks above now establish file availability and diagnostic population parity;
they do not establish numerical equivalence. For the original index audit, see
`snapshot_refresh_20261111/metadata/fresh_index_audit.json` under the data root.

For reproducibility or interrupted-run recovery, the completed extraction used:

```bash
python3 -B notebooks/espe_code/extract_argo_snapshot.py \
  --archive /swot/SUM05/amf2288/argo_mixing/snapshot_refresh_20261111/download/138320.tar.gz \
  --archive-sha256 029440622324ed0b003d851bf98cf6a2e0edc7e343a27af681c426c5d8b2cdc3 \
  --output /swot/SUM05/amf2288/argo_snapshots/gdac_20261109_138320 \
  --workers 4
```

The completed NATRE input audit used the following command. Its output already
exists and is preserved; choose a new report path for any deliberate rerun:

```bash
/home/amf2288/miniconda3/envs/Argo_Feb_25/bin/python -B \
  notebooks/espe_code/verify_argo_snapshot.py \
  --gdac /swot/SUM05/amf2288/argo_snapshots/gdac_20261109_138320 \
  --reference-run /swot/SUM05/amf2288/argo_mixing/natre_validation_20261110 \
  --reference-raw /swot/SUM05/amf2288/espe_code/data_files \
  --output /swot/SUM05/amf2288/argo_mixing/snapshot_refresh_20261111/natre_refreshed_input_audit.json \
  --workers 4
```

The downloader and extractor passed synthetic interruption/integrity/path tests;
the input verifier passed the six saved old/current profile controls. The archive
SHA256 establishes downloaded-source integrity. Extraction completion verifies
every core-index path, not a second independent hash of every extracted file.

**November 11 investigation update — read this before the November 10 handoff below.**

The user wants exact NATRE replication with an analysis that can scale globally,
and explicitly suggested testing unmodified published Argopy instead of the
editable checkout. The production script and completed validation outputs have
not been changed. Global authorization remains false. No full scientific rerun
or bulk archive refresh was performed during that initial investigation; the
subsequently authorized archive refresh is documented above.

New diagnostic reports and small downloaded controls are under:
`/swot/SUM05/amf2288/argo_mixing/natre_investigation_20261111`.

The main findings are:

- **The physical-profile overlap is 4,980, not 692.** Matching by
  `(PLATFORM_NUMBER, CYCLE_NUMBER, DIRECTION)` finds every accepted local profile
  in the 6,261-profile reference, with no physical-identity duplicates or
  local-only profiles. The actual deficit is 1,281. This diagnostic matching
  does not change the strict numerical validation gate.
- **The 1,281 deficit comprises 820 research-QC rejections, 258 resolution
  rejections, 202 profiles absent from the local index, and one profile with
  missing local index coordinates.** Of the 820, 804 contain only R/A data modes
  and no finite adjusted-pressure error. The remaining 16 have POSITION_QC=8.
  A representative of those 16, float 6901474/cycle 217, has POSITION_QC=1 in the
  current GDAC file after an August 11, 2023 update, later than the local index.
  That control supports archive revision; all 16 have not been checked remotely.
- **All 258 resolution rejections expose a loading problem.** Each mono-cycle
  file contains a primary sampling profile and two retained secondary samples.
  Our current loader merges them in file order under one float/cycle/direction,
  appending shallow secondary samples after the deep primary profile. This
  creates artificial pressure jumps and mean spacing of 5.885–10.598 dbar.
  All 258 primary-only sequences pass, with spacing 1.970–2.322 dbar. Their
  unrounded primary pressure and temperature are bitwise identical to reference
  inputs for every one of the 258; salinity matches exactly for none. Sorting
  alone would retain extra measurements absent from the reference.
- **There are genuine backend representation differences on unchanged files.**
  Old and current GDAC `D1901694_210.nc` are byte-identical. A live ERDDAP control
  returns only its 505 primary measurements and reproduces the saved reference
  pressure, temperature, salinity, and whole-second timestamp exactly. Of its
  505 salinities, 316 differ from GDAC by one float32 ULP (up to
  3.814697265625e-6). A five-decimal rounding hypothesis fails at a halfway value;
  do not implement an assumed salinity conversion rule. The exact server rule
  remains unverified.
- **Timestamp differences have two causes.** All shared coordinates match
  exactly. For 4,896 shared profiles the time difference is at most about
  2.286 ms. The other 84, all from float 6901151, have local timestamps two days
  later than the reference. A current GDAC sample corrects that two-day offset
  while leaving a 256 ns decoding difference. Do not round away real revisions.
- **The user's Argopy edits are not the cause of the tested differences.**
  The editable checkout reports 1.0.0, revision
  `a2af40f717d2f8d306238ad59866fd008df8f49e` (`v1.0.0-50-ga2af40f7`). Its local
  changes concern fetcher/import compatibility, filtering files without PSAL,
  and unrelated Canyon code. The preprocessing/QC/conversion code is unmodified;
  our loader bypasses the modified GDAC fetcher. Official PyPI 1.0.0 and 1.4.0
  were tested in separate temporary environments. On six fixed old/current input
  files, both produced the same loaded arrays, times, and QC outcomes as the
  editable checkout. Neither resolves the sampling merge or pressure rounding.
  This is an ingestion comparison, not a full scientific validation with either
  release. A broader 38-profile sample spanning all ten years also gave identical
  recorded comparisons under all three installations: standard GDAC preprocessing
  reproduced pressure/temperature/salinity together for 18 of 38 reference
  profiles; isolated removal of pressure rounding increased this to 19 of 38.
  Removing rounding alone therefore does not establish parity.

Evidence files in the investigation directory:

| File | Purpose |
| --- | --- |
| `identity_audit.json` | Complete identity overlap and deficit classification |
| `missing_profile_metadata.json` | Metadata of the 1,079 locally present rejected/missing-position profiles |
| `multisampling_audit.json` | Actual-loader verification of all 258 sampling/spacing failures |
| `argopy_checkout_audit.json` | Editable checkout version and relevant diffs |
| `published_argopy_provenance.json` | Official wheel URLs and verified SHA256 hashes |
| `probe_argopy_editable_smoke.json` | Six-file editable-package control |
| `probe_argopy_1_0_0.json`, `probe_argopy_1_4_0.json` | Same six files with official releases |
| `published_release_comparison.json` | Equal six-file outputs, unchanged official package sources, and equal 38-profile comparison statistics |
| `ingestion_v2/ingestion_experiments.json` | 38 shared profiles, installed preprocessing versus isolated removal of pressure rounding |
| `published_1_0_0/ingestion_experiments.json`, `published_1_4_0/ingestion_experiments.json` | Same 38-profile experiment with official releases |
| `current_gdac_samples/downloads.json` | Three small current-GDAC downloads and checksums |
| `live_erddap_probe/comparison.json` | Unchanged GDAC file versus saved/live ERDDAP values |
| `qc_probe_6901474_217/comparison.json` | Representative QC=8 to QC=1 archive revision |

The top-level `ingestion_experiments.json` is an invalid preliminary diagnostic:
its identity mask compared float-converted numeric identifiers with strings and
selected zero points. Use the corrected `ingestion_v2` report; the diagnostic now
asserts that previously accepted identities remain nonempty. No production run
used the invalid diagnostic.

Reusable diagnostics live beside the production script:
`diagnose_argo_identity.py`, `diagnose_argo_inputs.py`,
`diagnose_argo_multisampling.py`, and `probe_argo_release.py`. These are diagnostic
utilities, not alternate scientific production pipelines. Their reports record
scope and provenance; the probes do not run mixing calculations.

Temporary package environments are under
`/tmp/argo-published-comparison-20261111/{venv10,venv14}` and can be rebuilt.
Both use Python 3.11 and inherit the scientific environment's installed packages.
The 1.0.0 comparison retains xarray 2023.12.0; 1.4.0 uses xarray 2025.9.0 and
aiohttp 3.12.15. Both use erddapy 2.2.1, because current erddapy 3.3.0 breaks
private imports in the unmodified releases. The erddapy 2.2.1 release was yanked
for Python 3.9 compatibility; these tests use Python 3.11. These are controlled
comparison environments, not a finalized global environment lock. Your working
environment and editable Argopy were left intact. Published 1.4.0 opens a cache
during import before `set_options` is available; the diagnostic redirects only
`~/.cache/argopy` expansion to `/tmp`, without changing HOME or release code.

**Recommended direction:** use a pinned published package and a frozen input
snapshot. Argo officially supports monthly DOI snapshots and local mirrors;
server queries are also feasible in bounded chunks that are saved and checked
for completeness. Live queries alone do not make a fixed historical population
reproducible. Preserve the saved NATRE ERDDAP inputs as the exact regression
fixture. If the global source must also reproduce those inputs bit-for-bit,
either establish the ERDDAP representation rules completely or acquire and freeze
ERDDAP-format chunks. A newer GDAC snapshot by itself does not resolve the
backend differences demonstrated above. Do not replace the baseline or relax
equality without an explicit decision.

Official sources:
[Argo GDAC access and DOI snapshots](https://argo.ucsd.edu/data/data-from-gdacs/),
[Argo data revision FAQ](https://argo.ucsd.edu/data/data-faq/),
[sampling-profile definitions](https://oneargo.github.io/argo-format-user-manual/09-tables-reference.html#reference-table-16-vertical-sampling-schemes),
[Argopy parallel acquisition](https://argopy.readthedocs.io/en/latest/advanced-tools/performances/parallelization.html),
[published Argopy](https://pypi.org/project/argopy/).

Next work should address sampling selection and source provenance explicitly,
then test full-population equivalence in a new output directory. Historical
bootstrap realizations remain unrecoverable bit-for-bit without the original RNG
state; same-input, same-seed algorithm equivalence is the reproducible target.
All diagnostic jobs completed. No investigation process remains running.

**Original November 10 handoff:**

The NATRE processing and comparison work is complete. No processing jobs or Dask workers from this work remain running; closing the remote connection will not interrupt an unfinished calculation. The completed run is named `natre_validation_20261110`. Global processing has **not** started and remains blocked by the failed local-versus-reference comparison.

Start the next session by reading this file and the saved `validation.json`, then inspect the existing outputs before rerunning anything. The work and results are saved on the remote filesystem, so reconnecting is sufficient; nothing depends on keeping this chat or a notebook kernel alive.

The user's requirements remain:

- Put all scientific processing in the Python script. A future notebook should read saved results and visualize them.
- Accept spatial and time constraints; support the global archive and all locally available dates.
- Use the local Argo archive and the user's local Argopy installation, with bounded parallel processing.
- First test NATRE for **2014–2023**, and compare against the existing workflow. Proceed globally only after the required equivalence check passes.
- **No datasets in the home directory.** Put datasets, reports, logs, and Dask spill in new subdirectories under `/swot/SUM05/amf2288`. Repository code and this requested handoff document live in `notebooks/espe_code`; small runtime caches may use `/tmp`.

The new implementation is [process_argo_mixing.py](process_argo_mixing.py). It contains the profile calculations, the original notebook's regional analysis and bootstrap calculations, and its numerical plotting products. It writes separate profile partitions and regional summary NetCDFs. It includes worker-side writes, bounded pending Dask tasks, per-batch accounting, restart fingerprints, scientific failure flags, and a gate for global/large requests.

The existing source files are [runs_all_years_gamman.py](runs_all_years_gamman.py) and [plots_mixing_NATRE_gamman_analysis.ipynb](plots_mixing_NATRE_gamman_analysis.ipynb). They were preserved. No new visualization notebook has been created yet; the latest implementation request was to start with the processing script.

Useful paths:

| Item | Location |
| --- | --- |
| Repository | `/home/amf2288/argo-intern` |
| Working Python | `/home/amf2288/miniconda3/envs/Argo_Feb_25/bin/python` |
| Editable Argopy | `/home/amf2288/argopy/argopy` |
| Editable neutral-density package | `/home/amf2288/neutral_density/src` |
| Local GDAC archive | `/swot/SUM05/dbalwada/Argo_sync` |
| Historical raw and processed NATRE files | `/swot/SUM05/amf2288/espe_code/data_files` |
| Historical regional summary | `/swot/SUM05/amf2288/espe_code/NATRE_Argo_mixing_gamman.nc` |
| Completed run directory | `/swot/SUM05/amf2288/argo_mixing/natre_validation_20261110` |
| Validation report | `/swot/SUM05/amf2288/argo_mixing/natre_validation_20261110/validation.json` |
| **Complete local regional summary** | `/swot/SUM05/amf2288/argo_mixing/natre_validation_20261110/local/summaries/region.nc` |
| Local profile partitions | `/swot/SUM05/amf2288/argo_mixing/natre_validation_20261110/local/profiles/region/*.nc` |
| Reference-input replay partitions | `/swot/SUM05/amf2288/argo_mixing/natre_validation_20261110/legacy_raw/profiles/region/*.nc` |

Each profile-stage directory has `manifest.json`, `inventory.json`, and per-batch JSON status files. Both manifests now have `status: complete` and no failed batches. The initial `run.log` records a failure that was subsequently repaired; consult `validation.json`, `finish.log`, `local_summary.log`, and `final_check.log` for the final outcome.

The NATRE selection is longitude **−38 to −20**, latitude **19 to 32**, and UTC time **2014-01-01 inclusive to 2024-01-01 exclusive**. Spatial selection uses half-open upper bounds, except the global northern pole. No accepted historical reference profiles were lost at the NATRE upper boundaries. A preliminary inclusive index count was 13,030; the final production inventory selected **13,029 files**, as recorded in its manifest.

The completed tests established the following:

- **Reference-input profile replay passed exactly.** Reprocessing the saved research-mode raw inputs reproduced all **34 historical scientific variables across 6,261 profiles**, including NaN masks and coordinates, with maximum absolute error **0**. This took 169 batches. Duplicate identity occurrences were preserved.
- **Full-reference deterministic regional calculations passed exactly.** All 17 deterministic saved variables plus the two coordinates matched the historical summary on all 6,261 reference profiles.
- **Seeded bootstrap regression passed.** An independent execution of the original notebook's scientific cells and the new implementation matched on 100 profiles spanning 2014, using 1,000 bootstrap replicates and seed `20261110`. The historical notebook did not save its random seed; its original stochastic realization cannot be reproduced bitwise merely by rerunning it.
- **The local archive was fully processed for the requested selection.** All 204 batches completed, yielding **4,980 accepted profiles**. The complete local regional summary was also calculated with 1,000 bootstrap replicates: **223 data variables, 40 time windows, and 26 density bins**, plus depth/profile dimensions and plotting products. It is a complete local result, not the small validation sample.
- **Local-versus-historical equivalence failed.** The report is `mismatch_global_blocked`, with `passed: false` and `global_authorized: false`. The complete local regional summary was also compared against the historical summary and does not match.

Do not confuse these additional validation artifacts with the complete local result:

- `reference_deterministic_check.nc` contains the full-reference deterministic check. Bootstrap-dependent fields were deliberately omitted.
- `seeded_summary_subset.nc` contains the **100-profile validation subset**, including all plotting products. It is not the full NATRE analysis.
- Full-population, same-input, seeded bootstrap equivalence was **not** rerun after the local profile comparison failed. The full local summary itself **was** calculated; these are different checks.

The input mismatch needs investigation before global processing. The local index is dated **August 4, 2023**, and its last selected entry is `2023-08-04T09:50:20`. Historical raw reference files were fetched from ERDDAP on **November 8, 2026** and include profiles through December 2023. There are also differences in timestamps, pressure representation, salinity, and retained levels/QC selection.

In particular, the installed Argopy GDAC preprocessing rounds pressure, including adjusted pressure, to 0.1 dbar. Sample local-versus-reference differences included about 0.05 dbar in pressure, several millionths in salinity, and approximately millisecond timestamp offsets. Preserve the distinction between archive changes and backend representation changes; refreshing the archive alone may not resolve every numerical mismatch.

The comparison matched only **692 exact `(time_ns, longitude, latitude)` keys**. Its 5,569 unmatched reference keys and 4,288 unmatched local keys must **not** be described as that many distinct missing/new physical profiles: small timestamp offsets can prevent a match for the same physical profile. Nevertheless, 32 scientific variables differed even among exact-key matches, and the population/coverage discrepancy is real. Rounding timestamps or comparing only the intersection must not silently turn the full gate into a pass.

Implementation details worth preserving:

- The local loader directly opens indexed local files and applies the installed Argopy preprocessing/research QC. It does not construct a remote data fetcher. The earlier loader's `ftp=` argument did not match the local checkout's `gdac=` interface.
- Batch `part_0000156` initially failed because 16 real-time files had blank `SCIENTIFIC_CALIB_DATE` metadata. The final loader drops that unused history field before Argopy casting. Control tests showed identical scientific/QC arrays; the offending files then reached their correct research-QC rejection. The batch was repaired without discarding other completed results.
- Constant/single-level variables, unavailable neutral-density calculations, sparse profiles, and invalid bootstrap bins have explicit handling. The notebook's malformed insufficient-data return tuple was repaired without changing valid-bin results.
- Scientific defaults retain research QC, mean pressure spacing no greater than 5 dbar above 1,000 dbar, 2 m depth averaging to 2,000 m, 200 m finescale windows, 3-month output spacing, and 12-month averaging windows. “All profiles” means all matching candidates considered; quality/resolution exclusions remain.
- Known legacy conventions are deliberately preserved for parity: `CT_from_t` receives practical salinity, `SA_fs` stores `CT_fs`, the polynomial salinity residual bootstrap uses temperature `chi`, negative residual production is floored, and density windows overlap. These are recorded in output attributes. Scientific corrections should be explicit and separately validated, not introduced silently.
- The density range and 4 m correlation-length assumption remain NATRE defaults. Regional size, density coverage, equatorial/neutral-density limits, and uncertainty assumptions need review before interpreting global results.
- Regional aggregation was accelerated by selecting time-window columns before density calculations. Full-reference deterministic results stayed identical; the measured check fell from about 498 to 62 seconds.
- The final script also batches bootstrap draws, preserving draw order, arithmetic, and RNG state. It passed 2,351 synthetic cases, post-cleanup checks, and an integrated comparison of all 223 saved variables on the seeded 100-profile subset. That final integrated check took about 7.6 seconds on this machine.

Version provenance matters when resuming. The full local summary took about 1,220 seconds with the scalar bootstrap helpers. It was saved before the final batched optimization; the entire 4,980-profile summary has **not** been recomputed with the final batched helpers. Their equivalence was checked separately as described above. The report retains the producing version and the final tested script version:

```text
Final script SHA256:
7b3edfe77dcc7af6d085e20c1d92c07785d35f62abb9f21035e01136b88bd3e8

Full local summary producing-script SHA256:
c7affea69c87ce5c1647bf972cadc21d3e77b95484a7305cb9b858a09aeee861
```

The earlier profile kernels were verified unchanged across the later source revisions. Existing profile manifests retain their historical fingerprints and recovery provenance. **Use a fresh, explicit output directory for a new run after changing code, inputs, or scientific configuration.** Do not overwrite or casually modify completed manifests to bypass fingerprint checks. The CLI's default output is `.../argo_mixing/natre_validation`, not the completed `.../natre_validation_20261110` directory.

Suggested next-session sequence:

1. Read this handoff and `validation.json`; open the complete local summary and inspect coverage and comparisons. The completed run does not need to be restarted just because the remote connection was closed.
2. Diagnose input equivalence using float/cycle/direction identities from the saved raw inputs and new profile outputs. Separate timestamp/coordinate representation effects from archive coverage, QC, and adjusted-value differences.
3. Discuss which reference population should govern the gate: a refreshed local mirror aligned with the historical reference, or a clearly identified new original-workflow reference computed from the same local snapshot. Do not automatically download a large replacement archive, relax equality, or redefine the baseline without settling that choice with the user.
4. Resolve and document the input differences, then validate in a fresh output directory. Preserve any genuine scientific corrections as explicit changes with separate comparisons.
5. Only after the required gate passes, settle the global aggregation settings and proceed globally. A future visualization notebook should consume saved products rather than redo processing.

Read the existing full summary in the `Argo_Feb_25` environment, keeping incidental caches out of the home directory:

```python
import os
os.environ['MPLCONFIGDIR'] = '/tmp/argo-mixing-runtime/matplotlib'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/argo-mixing-runtime/numba'
os.environ['XDG_CACHE_HOME'] = '/tmp/argo-mixing-runtime/cache'

import xarray as xr
ds = xr.open_dataset(
    '/swot/SUM05/amf2288/argo_mixing/natre_validation_20261110/local/summaries/region.nc',
    engine='netcdf4',
)
print(ds)
```

For help, without starting processing:

```bash
cd /home/amf2288/argo-intern
/home/amf2288/miniconda3/envs/Argo_Feb_25/bin/python -B \
  notebooks/espe_code/process_argo_mixing.py --help
```

When a new validation is needed after resolving the loader/input issues above,
use the refreshed archive and a new destination (example updated November 11):

```bash
/home/amf2288/miniconda3/envs/Argo_Feb_25/bin/python -B \
  notebooks/espe_code/process_argo_mixing.py validate-natre \
  --gdac /swot/SUM05/amf2288/argo_snapshots/gdac_20261109_138320 \
  --output /swot/SUM05/amf2288/argo_mixing/natre_validation_next \
  --workers 12 --batch-size 64 --memory-limit 8GiB
```

`validate-natre` fixes the NATRE bounds and 2014–2023 dates itself. Exit code 2 means a completed comparison found a mismatch. `run` accepts custom bounds and dates; global/large requests require a passing report for the matching script/archive. Omitting dates means all dates present in the selected local index. Use `--cell-size 0` for one selected regional summary; the global tiling default is 10 degrees. A revalidation that fails at profile equivalence produces diagnostic summary checks; it does not automatically calculate the full local regional summary. The completed full summary above was calculated separately through the script's summary function.

The bare `python` and `rg` commands were unavailable in the session shell. Use the explicit environment Python for science; system `python3` suffices for reading JSON, and `grep`/`find` are available. `seawater` deprecation and optional `erddapy` backend warnings occurred but did not stop this workflow.

At handoff, `process_argo_mixing.py` is a new, untracked file, and no commit has been made. `notebooks/argo_pbar/NATRE_terms.ipynb` already contained user changes before this work; leave them intact. Temporary implementation/test helpers under `/tmp` are disposable and must not be required to resume. The durable implementation is the single repository script, and the durable results and provenance are under `/swot`.
