# Peakachu
A Lightweight and Limited mzML Viewer

A single-file [marimo](https://marimo.io) notebook for loading an mzML file,
inspecting the TIC and mass spectra, defining m/z extraction windows, and
pulling integrated intensities across those windows — for one file interactively
or many files in batch.

The notebook is **self-contained**: the indexed mzML reader is embedded directly
in the first cell, so there are no sibling-module imports. That is what makes the
WASM/Pyodide export work (see below).

## Run locally (server mode)

```bash
pip install marimo numpy pandas plotly
marimo run mzml_window_viewer.py
```

Or open it as an editable notebook:

```bash
marimo edit mzml_window_viewer.py
```

## Deploy to the web (Pyodide / WASM, e.g. GitHub Pages)

```bash
marimo export html-wasm mzml_window_viewer.py -o site --mode run
```

Then serve the `site/` directory over HTTP (WASM will not run from a `file://`
path). For a quick local check:

```bash
python -m http.server -d site 8000
```

## Using it

- **Load** tab — two ways to open a file:
  - *Upload* (`.mzML`) — works in every deployment, including the hosted WASM
    build. Capped at **100 MB** (a marimo limit) and held in browser memory.
  - *Pick a file on disk* — shown only when running with `marimo run`/`marimo
    edit`. Hands the reader a path; spectra are read by byte range straight from
    disk, so there is **no size limit** and the whole file is never loaded into
    memory. Use this for large acquisitions.
- **Explore & windows** tab — view the TIC (blue) and a mass spectrum (orange);
  pick a scan, an RT range, or sum all scans. Define m/z windows in the editor;
  they are shaded on the spectrum and integrated in the stats table. Windows
  **persist across files** loaded in the same session. Save the current windows
  to JSON and load them back later. **Export spectrum (CSV)** saves the
  displayed spectrum as `mz, intensity`, optionally clipped to one window.

  To grab a window straight off the plot, set **Spectrum interaction** to
  *Select an m/z window*, drag a box across the region of interest, and press
  **Add current selection** — the span is appended to the window table.
- **Batch extract** tab — select many `.mzML` files, then extract the integrated
  intensity of every active window from every file into one table, downloadable
  as CSV.

## A note on file size (mo.ui.file limit)

`mo.ui.file` uploads have a hard **100 MB** ceiling, and oversized uploads can
fail silently. This is a real constraint for large profile-mode acquisitions.
The workaround built into the notebook is to run it in server mode and use the
on-disk file browser instead of uploading: because the reader seeks by byte
offset, it streams from the original file with no size limit and no whole-file
memory cost. In a static WASM export there is no server filesystem, so upload is
the only option there — for very large files, run locally with `marimo run`.

## Files

- `mzml_window_viewer.py` — the app (self-contained; this is all you need).
- `mzml_indexed_reader.py` — standalone copy of the reader, in case you want to
  reuse it in other scripts. The notebook does not import it.

## Note on "add current zoom"

marimo's Plotly integration reports *selection* ranges (box/lasso) but not
zoom/relayout ranges — a pan/zoom viewport is never sent back to Python. So the
capture button reads the **box-selected** span rather than the zoom viewport.
It's the same drag gesture, and it's more precise, since you state the range
explicitly instead of inferring it from the axis limits.

