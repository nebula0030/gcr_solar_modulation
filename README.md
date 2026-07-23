# Muon Rate vs. Solar Activity

Turns a CosmicWatch v3X data file into a pressure/temperature-corrected muon
rate and compares it against public solar-activity data on interactive plots.

## Setup

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

```bash
.venv/bin/python process_data.py DATA_FILE --bin-length SECONDS [options]
```

Example:

```bash
.venv/bin/python process_data.py ../CW_NebuLab_004_2026-07-10_04-16-31.txt \
  --bin-length 3600 --export-csv
```

This writes `<run>_overlay.html` and `<run>_sidebyside.html`.

On a real ~2.74-day run from this detector (747,823 events parsed, 111,538
of them coincident, mean coincident rate ~0.47 Hz) with 1-hour bins, this
produced 65 complete bins, a mean Poisson error of 2.43% per bin, and a
fitted `beta_P` of -0.54 +/- 0.49 %/hPa — flagged with a narrow-pressure-range
warning because the run's barometric pressure only spanned 2.6 hPa. All four
external sources (NMDB, GOES, Kp, sunspot number) were retrieved successfully
for that run.

## How it works

1. **Parse.** Reads the 13-column v3X format. Timestamps are treated as UTC.
2. **Bin.** Counts only coincident (`Flag == 1`) events. Livetime per bin is
   the bin width minus the increase in the detector's cumulative deadtime
   counter.
3. **Correct.** Applies `R_corr = R * exp(-beta_P*(P-P0) - beta_T*(T-T0))`,
   with coefficients either fitted from the run or supplied.
4. **Fetch.** Pulls NMDB neutron monitor counts, GOES X-ray flux, Kp index,
   and sunspot number for the run's window, cached locally.
5. **Align.** Resamples everything onto the rate bins, flagging interpolated
   points.
6. **Plot.** Emits both an overlay and a linked-axis side-by-side view.

## Choosing a bin length

Poisson error per bin at ~0.47 Hz:

| Bin length | Fractional error |
| --- | --- |
| 1 hour | 2.4% |
| 6 hours | 1.0% |
| 24 hours | 0.5% |

These are the values actually measured on a real run: 65 complete 1-hour
bins gave a mean per-bin error of 2.43%, and 10 complete 6-hour bins gave
0.99%. The 24-hour figure follows the same 1/sqrt(t) scaling.

Solar modulation in ground-level muon rate is typically 0.5-3%, so bins
shorter than about an hour bury the signal in counting noise.

## Two things to be aware of

**The temperature correction is a detector systematic, not the atmospheric
temperature effect.** The onboard BMP280 measures enclosure temperature,
which tracks SiPM gain drift and therefore the effective trigger threshold.
The classical atmospheric temperature effect depends on upper-air
temperature at the pion-production altitude and is not corrected here.

**`--correction-method literature` has no default coefficient.** No
verifiable CosmicWatch-specific published barometric coefficient was found,
so you must pass `--beta-p` explicitly and are encouraged to record where it
came from via `--beta-p-source`. Use `--correction-method fit` on a long run
to derive your own.

## Station selection

The comparison neutron monitor is chosen by matching geomagnetic cutoff
rigidity, not geographic distance, because rigidity governs which primaries
reach a location and therefore the size of solar modulation signals. Note
that this tends to favour high-altitude mountain stations; override with
`--nmdb-station` for a sea-level comparison.

The rigidity-best match is not always the station that gets used: NMDB does
not have data for every station at every moment, and coverage gaps are
common for the small, high-altitude stations that rigidity matching tends
to favour. If the top-ranked station has no usable data for the run's
window, the tool automatically tries the next-best rigidity match, and the
next, exhausting the full ranking if needed, until it finds a station that
actually has data. Each station it rules out is reported as a `NOTE` in the
console summary (e.g. "station UFSZ has no data for this run window ...
trying next-best rigidity match"), along with which station it ultimately
fell back to. This fallback is expected behaviour, not a failure — but it
means the exact station used can vary run to run as NMDB's live data
coverage changes, even for the same detector location and the same
requested bin length.

## Options

Run `.venv/bin/python process_data.py --help` for the full list.

## Data sources

- Neutron monitors: [NMDB](https://www.nmdb.eu/)
- X-ray flux: [NOAA SWPC](https://services.swpc.noaa.gov/) and
  [NOAA NCEI](https://www.ngdc.noaa.gov/)
- Kp index: [GFZ Potsdam](https://kp.gfz.de/)
- Sunspot number: [SILSO, Royal Observatory of Belgium](https://www.sidc.be/SILSO/)

We acknowledge the NMDB database (www.nmdb.eu), founded under the European
Union's FP7 programme (contract no. 213007), and the PIs of the individual
neutron monitor stations.
