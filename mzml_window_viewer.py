# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.20",
#     "numpy>=1.26",
#     "pandas>=2.0",
#     "plotly>=5.20",
# ]
# ///
"""Peakachu - a browser-ready mzML window-intensity viewer (marimo + Pyodide).

Files are uploaded through the browser (``mo.ui.file``) and written to the
in-memory temp filesystem, so the same notebook runs both with ``marimo run``
and as a static WASM export. No native OS file dialogs or subprocesses are used.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full", app_title="Peakachu")


@app.cell
def _():
    # =====================================================================
    # Embedded indexed-mzML reader (self-contained for Pyodide/WASM deploy).
    # Random-access, disk-backed decoding; only the compact index + TIC and
    # one spectrum are resident at a time.
    # =====================================================================
    from dataclasses import dataclass
    from pathlib import Path
    import base64
    import mmap
    import re
    import zlib
    from typing import Callable, Iterable, Iterator
    import xml.etree.ElementTree as ET

    import numpy as np


    class MzMLIndexedReaderError(RuntimeError):
        """Raised when an mzML file cannot be indexed or decoded."""


    @dataclass(frozen=True)
    class SpectrumRecord:
        """One decoded spectrum and its compact metadata."""

        index: int
        scan_number: int
        spectrum_id: str
        time_min: float
        ms_level: int
        mode: str
        tic: float
        mz: np.ndarray
        intensity: np.ndarray


    @dataclass(frozen=True)
    class SumResult:
        """A streamed summed spectrum on a common grid.

        ``bin_area`` stores the area-conserving signal assigned to each output bin.
        For profile data, ``intensity`` is the corresponding intensity density
        (``bin_area / bin_width``), which is appropriate for plotting.
        """

        mz: np.ndarray
        intensity: np.ndarray
        bin_edges: np.ndarray
        bin_area: np.ndarray
        requested_count: int
        included_count: int
        first_index: int | None
        last_index: int | None
        ms_level: int | None
        method: str
        grid_mode: str
        sampling_resolution: float | None
        oversampling: float | None


    _CV_TAG_RE = re.compile(rb"<cvParam\b[^>]*?/?>", re.IGNORECASE)
    _ATTR_RE = re.compile(rb"([A-Za-z_:][\w:.-]*)\s*=\s*([\"'])(.*?)\2", re.DOTALL)
    _OFFSET_RE = re.compile(
        rb"<offset\b[^>]*\bidRef\s*=\s*([\"'])(.*?)\1[^>]*>\s*(\d+)\s*</offset>",
        re.IGNORECASE | re.DOTALL,
    )


    def _attrs(tag: bytes) -> dict[str, str]:
        result: dict[str, str] = {}
        for match in _ATTR_RE.finditer(tag):
            key = match.group(1).decode("utf-8", "replace")
            value = match.group(3).decode("utf-8", "replace")
            result[key] = value
        return result


    def _cv_params_bytes(fragment: bytes) -> list[dict[str, str]]:
        return [_attrs(match.group(0)) for match in _CV_TAG_RE.finditer(fragment)]


    def _param_value(
        params: Iterable[dict[str, str]], accession: str, default: str | None = None
    ) -> str | None:
        for param in params:
            if param.get("accession") == accession:
                return param.get("value", default)
        return default


    def _has_param(params: Iterable[dict[str, str]], accession: str) -> bool:
        return any(param.get("accession") == accession for param in params)


    def _scan_number(identifier: str, fallback: int) -> int:
        match = re.search(r"(?:^|\s)scan=(\d+)", identifier)
        return int(match.group(1)) if match else int(fallback)


    def _time_to_minutes(value: float, unit_accession: str | None, unit_name: str | None) -> float:
        unit_accession = unit_accession or ""
        unit_name = (unit_name or "").lower()
        if unit_accession == "UO:0000010" or unit_name.startswith("second"):
            return value / 60.0
        if unit_accession == "UO:0000032" or unit_name.startswith("hour"):
            return value * 60.0
        return value


    def _local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]


    def _element_params(
        element: ET.Element,
        reference_groups: dict[str, list[dict[str, str]]],
    ) -> list[dict[str, str]]:
        params: list[dict[str, str]] = []
        for child in element.iter():
            name = _local_name(child.tag)
            if name == "cvParam":
                params.append(dict(child.attrib))
            elif name == "referenceableParamGroupRef":
                params.extend(reference_groups.get(child.attrib.get("ref", ""), ()))
        return params


    def _decode_binary_array(
        array_element: ET.Element,
        reference_groups: dict[str, list[dict[str, str]]],
    ) -> tuple[str | None, np.ndarray]:
        params = _element_params(array_element, reference_groups)

        if _has_param(params, "MS:1000514"):
            kind = "mz"
        elif _has_param(params, "MS:1000515"):
            kind = "intensity"
        elif _has_param(params, "MS:1000595"):
            kind = "time"
        else:
            kind = None

        if _has_param(params, "MS:1000523"):
            dtype = np.dtype("<f8")
        elif _has_param(params, "MS:1000521"):
            dtype = np.dtype("<f4")
        elif _has_param(params, "MS:1000522"):
            dtype = np.dtype("<i8")
        elif _has_param(params, "MS:1000519"):
            dtype = np.dtype("<i4")
        else:
            raise MzMLIndexedReaderError("Unsupported mzML binary numeric type")

        binary_text = ""
        for child in array_element.iter():
            if _local_name(child.tag) == "binary":
                binary_text = child.text or ""
                break

        try:
            payload = base64.b64decode("".join(binary_text.split()), validate=False)
            if _has_param(params, "MS:1000574"):
                payload = zlib.decompress(payload)
            elif not _has_param(params, "MS:1000576"):
                # Numpress and other encodings are deliberately not guessed.
                compression_accessions = {
                    p.get("accession", "") for p in params if p.get("accession", "").startswith("MS:10023")
                }
                if compression_accessions:
                    raise MzMLIndexedReaderError(
                        "MS-Numpress-compressed arrays are not supported by this lightweight reader"
                    )
            values = np.frombuffer(payload, dtype=dtype).astype(float, copy=True)
        except MzMLIndexedReaderError:
            raise
        except Exception as exc:
            raise MzMLIndexedReaderError(f"Could not decode an mzML binary array: {exc}") from exc

        return kind, values


    def _parse_xml_fragment(fragment: bytes, expected: str) -> ET.Element:
        start = fragment.find(f"<{expected}".encode())
        end_token = f"</{expected}>".encode()
        end = fragment.rfind(end_token)
        if start < 0 or end < 0:
            raise MzMLIndexedReaderError(f"Could not isolate <{expected}> XML fragment")
        end += len(end_token)
        try:
            return ET.fromstring(fragment[start:end])
        except ET.ParseError as exc:
            raise MzMLIndexedReaderError(f"Malformed {expected} XML fragment: {exc}") from exc


    def _prepare_spectrum_arrays(
        mz: np.ndarray,
        intensity: np.ndarray,
        *,
        duplicate_relative_tolerance: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sort, clean, and combine duplicate or nearly duplicate m/z coordinates."""
        mz = np.asarray(mz, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        size = min(mz.size, intensity.size)
        mz = mz[:size]
        intensity = intensity[:size]
        valid = np.isfinite(mz) & np.isfinite(intensity)
        if not np.any(valid):
            return np.array([], dtype=float), np.array([], dtype=float)
        mz = mz[valid]
        intensity = intensity[valid]
        order = np.argsort(mz, kind="mergesort")
        mz = mz[order]
        intensity = intensity[order]
        if mz.size < 2:
            return mz, intensity

        # Thermo-derived mzML can contain coordinates that differ only at the
        # floating-point-noise level. Treat those as one stored bin.
        diffs = np.diff(mz)
        mids = 0.5 * (mz[:-1] + mz[1:])
        tolerance = np.maximum(1e-12, duplicate_relative_tolerance * np.maximum(mids, 1.0))
        new_group = np.r_[True, diffs > tolerance]
        groups = np.cumsum(new_group) - 1
        if groups[-1] + 1 == mz.size:
            return mz, intensity

        count = np.bincount(groups)
        mz_sum = np.bincount(groups, weights=mz)
        intensity_sum = np.bincount(groups, weights=intensity)
        combined_mz = mz_sum / np.maximum(count, 1)
        return combined_mz.astype(float), intensity_sum.astype(float)


    def _typical_relative_spacing(mz: np.ndarray) -> float | None:
        """Estimate the native relative m/z spacing while rejecting large gaps."""
        mz = np.asarray(mz, dtype=float)
        if mz.size < 3:
            return None
        diff = np.diff(mz)
        mid = 0.5 * (mz[:-1] + mz[1:])
        rel = diff / np.maximum(mid, np.finfo(float).tiny)
        rel = rel[np.isfinite(rel) & (rel > 1e-12)]
        if rel.size == 0:
            return None
        median = float(np.median(rel))
        # Remove residual near-duplicates and omitted-region gaps, then recompute.
        core = rel[(rel >= median * 1e-3) & (rel <= median * 10.0)]
        if core.size:
            median = float(np.median(core))
        return median if np.isfinite(median) and median > 0 else None


    def _constant_resolution_edges(
        mz_min: float,
        mz_max: float,
        resolution: float,
        oversampling: float,
        max_grid_points: int,
    ) -> np.ndarray:
        if mz_min <= 0:
            raise ValueError("Automatic constant-resolution grids require m/z minimum > 0")
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("The estimated sampling resolution is invalid")
        if not np.isfinite(oversampling) or oversampling <= 0:
            raise ValueError("The oversampling factor must be greater than zero")
        grid_resolution = float(resolution) * float(oversampling)
        dlog = np.log1p(1.0 / grid_resolution)
        bin_count = int(np.ceil(np.log(mz_max / mz_min) / dlog))
        if bin_count < 1:
            raise ValueError("The automatic common grid contains no bins")
        if bin_count > int(max_grid_points):
            raise ValueError(
                f"The automatic common grid has {bin_count:,} bins. Reduce the "
                "oversampling factor or narrow the m/z limits."
            )
        edges = mz_min * np.exp(np.arange(bin_count + 1, dtype=float) * dlog)
        edges[-1] = mz_max
        return edges


    def _fixed_da_edges(
        mz_min: float,
        mz_max: float,
        step: float,
        max_grid_points: int,
    ) -> np.ndarray:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("The fixed m/z bin width must be greater than zero")
        bin_count = int(np.ceil((mz_max - mz_min) / step))
        if bin_count < 1:
            raise ValueError("The fixed common grid contains no bins")
        if bin_count > int(max_grid_points):
            raise ValueError(
                f"The requested fixed grid has {bin_count:,} bins. Increase the "
                "bin width or narrow the m/z limits."
            )
        edges = mz_min + np.arange(bin_count + 1, dtype=float) * step
        edges[-1] = mz_max
        return edges


    def _profile_segments(mz: np.ndarray, gap_factor: float) -> list[tuple[int, int]]:
        """Return contiguous profile regions without bridging omitted-data gaps."""
        if mz.size == 0:
            return []
        if mz.size == 1:
            return [(0, 1)]
        typical = _typical_relative_spacing(mz)
        if typical is None:
            return [(0, mz.size)]
        diff = np.diff(mz)
        mid = 0.5 * (mz[:-1] + mz[1:])
        rel = diff / np.maximum(mid, np.finfo(float).tiny)
        breaks = np.flatnonzero(rel > float(gap_factor) * typical) + 1
        starts = np.r_[0, breaks]
        stops = np.r_[breaks, mz.size]
        return [(int(a), int(b)) for a, b in zip(starts, stops) if b > a]


    def _deposit_profile_area(
        accumulator: np.ndarray,
        target_edges: np.ndarray,
        mz: np.ndarray,
        intensity: np.ndarray,
        *,
        gap_factor: float,
    ) -> None:
        """Deposit native profile-bin area into overlapping target bins.

        Each stored profile point is treated as the intensity density of its local
        Voronoi interval. This corrects for variable native m/z spacing and avoids
        turning sparse profile samples into delta-like spikes.
        """
        mz, intensity = _prepare_spectrum_arrays(mz, intensity)
        if mz.size == 0:
            return
        typical_rel = _typical_relative_spacing(mz)
        for start, stop in _profile_segments(mz, gap_factor):
            smz = mz[start:stop]
            sint = intensity[start:stop]
            if smz.size == 1:
                rel = typical_rel if typical_rel is not None else 1e-4
                half = max(smz[0] * rel * 0.5, np.finfo(float).eps)
                native_edges = np.array([smz[0] - half, smz[0] + half], dtype=float)
            else:
                mids = 0.5 * (smz[:-1] + smz[1:])
                native_edges = np.empty(smz.size + 1, dtype=float)
                native_edges[1:-1] = mids
                native_edges[0] = smz[0] - 0.5 * (smz[1] - smz[0])
                native_edges[-1] = smz[-1] + 0.5 * (smz[-1] - smz[-2])

            for left, right, value in zip(native_edges[:-1], native_edges[1:], sint):
                if not np.isfinite(value) or right <= target_edges[0] or left >= target_edges[-1]:
                    continue
                left = max(float(left), float(target_edges[0]))
                right = min(float(right), float(target_edges[-1]))
                if right <= left:
                    continue
                first = int(np.searchsorted(target_edges, left, side="right") - 1)
                last = int(np.searchsorted(target_edges, right, side="left"))
                first = max(0, min(first, accumulator.size - 1))
                last = max(first + 1, min(last, accumulator.size))
                for target_index in range(first, last):
                    overlap = min(right, float(target_edges[target_index + 1])) - max(
                        left, float(target_edges[target_index])
                    )
                    if overlap > 0:
                        accumulator[target_index] += float(value) * overlap


    def _deposit_points(
        accumulator: np.ndarray,
        target_edges: np.ndarray,
        mz: np.ndarray,
        intensity: np.ndarray,
    ) -> None:
        """UniDec-like point integration for centroid or comparison use."""
        mz, intensity = _prepare_spectrum_arrays(mz, intensity)
        if mz.size == 0:
            return
        bins = np.searchsorted(target_edges, mz, side="right") - 1
        keep = (bins >= 0) & (bins < accumulator.size)
        if np.any(keep):
            np.add.at(accumulator, bins[keep], intensity[keep])


    def _deposit_legacy_interpolation(
        density_accumulator: np.ndarray,
        target_centers: np.ndarray,
        mz: np.ndarray,
        intensity: np.ndarray,
        *,
        gap_factor: float,
    ) -> None:
        """Segmented linear interpolation retained for legacy comparison."""
        mz, intensity = _prepare_spectrum_arrays(mz, intensity)
        if mz.size == 0:
            return
        for start, stop in _profile_segments(mz, gap_factor):
            smz = mz[start:stop]
            sint = intensity[start:stop]
            if smz.size == 1:
                idx = int(np.argmin(np.abs(target_centers - smz[0])))
                density_accumulator[idx] += sint[0]
                continue
            left = int(np.searchsorted(target_centers, smz[0], side="left"))
            right = int(np.searchsorted(target_centers, smz[-1], side="right"))
            if right > left:
                density_accumulator[left:right] += np.interp(
                    target_centers[left:right], smz, sint
                )


    class IndexedMzMLSource:
        """Random-access, disk-backed mzML source.

        Parameters
        ----------
        path:
            Path to an uncompressed ``.mzML`` file.
        """

        def __init__(self, path: str | Path):
            self.path = Path(path).expanduser().resolve()
            if not self.path.exists() or not self.path.is_file():
                raise MzMLIndexedReaderError(f"mzML file does not exist: {self.path}")
            if self.path.suffix.lower() == ".gz":
                raise MzMLIndexedReaderError(
                    "Random access requires an uncompressed .mzML file; .mzML.gz is not supported"
                )
            if self.path.suffix.lower() != ".mzml":
                raise MzMLIndexedReaderError("Please select a file ending in .mzML")

            self.name = self.path.name
            self.file_size = int(self.path.stat().st_size)
            self.reference_groups: dict[str, list[dict[str, str]]] = {}

            self.spectrum_ids: tuple[str, ...] = ()
            self.scan_numbers = np.array([], dtype=np.int64)
            self._starts = np.array([], dtype=np.int64)
            self._ends = np.array([], dtype=np.int64)
            self._index_list_offset: int | None = None
            self._chromatogram_offsets: dict[str, int] = {}
            self.is_indexed = False

            self.times_min = np.array([], dtype=float)
            self.tic = np.array([], dtype=float)
            self._first_meta: dict[str, object] = {}

            self._build_offsets()
            self._read_reference_groups()
            self._load_tic_or_headers()
            self._first_meta = self._read_header_metadata(0)

        @property
        def n_scans(self) -> int:
            return int(self._starts.size)

        @property
        def time_min(self) -> float:
            return float(np.nanmin(self.times_min)) if self.times_min.size else 0.0

        @property
        def time_max(self) -> float:
            return float(np.nanmax(self.times_min)) if self.times_min.size else 0.0

        @property
        def first_ms_level(self) -> int:
            return int(self._first_meta.get("ms_level", 1))

        @property
        def first_mode(self) -> str:
            return str(self._first_meta.get("mode", "unknown"))

        @property
        def default_mz_min(self) -> float:
            for key in ("scan_window_low", "observed_low"):
                value = self._first_meta.get(key)
                if value is not None and np.isfinite(float(value)):
                    return float(value)
            return 0.0

        @property
        def default_mz_max(self) -> float:
            for key in ("scan_window_high", "observed_high"):
                value = self._first_meta.get(key)
                if value is not None and np.isfinite(float(value)):
                    return float(value)
            return 2000.0

        @property
        def compact_memory_bytes(self) -> int:
            return int(
                self.scan_numbers.nbytes
                + self._starts.nbytes
                + self._ends.nbytes
                + self.times_min.nbytes
                + self.tic.nbytes
                + sum(len(identifier.encode("utf-8")) for identifier in self.spectrum_ids)
            )

        def _build_offsets(self) -> None:
            with self.path.open("rb") as handle:
                tail_size = min(self.file_size, 2_000_000)
                handle.seek(self.file_size - tail_size)
                tail = handle.read(tail_size)

            matches = list(re.finditer(rb"<indexListOffset>\s*(\d+)\s*</indexListOffset>", tail))
            if matches:
                self._index_list_offset = int(matches[-1].group(1))
                self._build_from_index(self._index_list_offset)
                self.is_indexed = True
            else:
                self._build_from_mmap_scan()
                self.is_indexed = False

            if not self._starts.size:
                raise MzMLIndexedReaderError("No <spectrum> elements were found")

        def _build_from_index(self, index_offset: int) -> None:
            with self.path.open("rb") as handle:
                handle.seek(index_offset)
                index_bytes = handle.read()

            spectrum_match = re.search(
                rb"<index\b[^>]*\bname\s*=\s*([\"'])spectrum\1[^>]*>(.*?)</index>",
                index_bytes,
                re.IGNORECASE | re.DOTALL,
            )
            if not spectrum_match:
                raise MzMLIndexedReaderError("indexedmzML does not contain a spectrum index")

            identifiers: list[str] = []
            starts: list[int] = []
            for match in _OFFSET_RE.finditer(spectrum_match.group(2)):
                identifiers.append(match.group(2).decode("utf-8", "replace"))
                starts.append(int(match.group(3)))

            for index_match in re.finditer(
                rb"<index\b[^>]*\bname\s*=\s*([\"'])chromatogram\1[^>]*>(.*?)</index>",
                index_bytes,
                re.IGNORECASE | re.DOTALL,
            ):
                for match in _OFFSET_RE.finditer(index_match.group(2)):
                    identifier = match.group(2).decode("utf-8", "replace")
                    self._chromatogram_offsets[identifier] = int(match.group(3))

            starts_array = np.asarray(starts, dtype=np.int64)
            end_boundary = min(
                [index_offset, *self._chromatogram_offsets.values()]
                if self._chromatogram_offsets
                else [index_offset]
            )
            ends_array = np.empty_like(starts_array)
            if starts_array.size > 1:
                ends_array[:-1] = starts_array[1:]
            ends_array[-1] = end_boundary

            self.spectrum_ids = tuple(identifiers)
            self.scan_numbers = np.asarray(
                [_scan_number(identifier, i + 1) for i, identifier in enumerate(identifiers)],
                dtype=np.int64,
            )
            self._starts = starts_array
            self._ends = ends_array

        def _build_from_mmap_scan(self) -> None:
            starts: list[int] = []
            ends: list[int] = []
            identifiers: list[str] = []

            # Prefer a memory-mapped scan for large local files. In some sandboxed
            # runtimes (notably the Pyodide/WASM MEMFS used for browser deployment)
            # mmap is unavailable, so fall back to an in-memory byte scan.
            try:
                with self.path.open("rb") as handle:
                    buffer: "mmap.mmap | bytes" = mmap.mmap(
                        handle.fileno(), 0, access=mmap.ACCESS_READ
                    )
                    using_mmap = True
            except (OSError, ValueError):
                buffer = self.path.read_bytes()
                using_mmap = False

            try:
                position = 0
                while True:
                    start = buffer.find(b"<spectrum ", position)
                    if start < 0:
                        break
                    end = buffer.find(b"</spectrum>", start)
                    if end < 0:
                        raise MzMLIndexedReaderError("Unterminated <spectrum> element")
                    end += len(b"</spectrum>")
                    opening_end = buffer.find(b">", start, min(end, start + 8192))
                    opening = bytes(buffer[start : opening_end + 1]) if opening_end >= 0 else b""
                    identifier = _attrs(opening).get("id", f"spectrum={len(starts) + 1}")
                    starts.append(start)
                    ends.append(end)
                    identifiers.append(identifier)
                    position = end
            finally:
                if using_mmap:
                    buffer.close()

            self.spectrum_ids = tuple(identifiers)
            self.scan_numbers = np.asarray(
                [_scan_number(identifier, i + 1) for i, identifier in enumerate(identifiers)],
                dtype=np.int64,
            )
            self._starts = np.asarray(starts, dtype=np.int64)
            self._ends = np.asarray(ends, dtype=np.int64)

        def _read_reference_groups(self) -> None:
            first_start = int(self._starts[0])
            with self.path.open("rb") as handle:
                prefix = handle.read(first_start)
            try:
                wrapped = ET.fromstring(b"<root>" + prefix + b"</root>")
            except ET.ParseError:
                # The document preamble may include declarations that cannot be wrapped.
                group_list_start = prefix.find(b"<referenceableParamGroupList")
                group_list_end = prefix.rfind(b"</referenceableParamGroupList>")
                if group_list_start < 0 or group_list_end < 0:
                    return
                group_list_end += len(b"</referenceableParamGroupList>")
                try:
                    wrapped = ET.fromstring(prefix[group_list_start:group_list_end])
                except ET.ParseError:
                    return

            for element in wrapped.iter():
                if _local_name(element.tag) != "referenceableParamGroup":
                    continue
                group_id = element.attrib.get("id")
                if not group_id:
                    continue
                self.reference_groups[group_id] = [
                    dict(child.attrib)
                    for child in element.iter()
                    if _local_name(child.tag) == "cvParam"
                ]

        def _read_range(self, start: int, end: int) -> bytes:
            with self.path.open("rb") as handle:
                handle.seek(int(start))
                return handle.read(int(end) - int(start))

        def _read_spectrum_fragment(self, index: int) -> bytes:
            index = int(np.clip(index, 0, self.n_scans - 1))
            return self._read_range(int(self._starts[index]), int(self._ends[index]))

        def _read_header_metadata(self, index: int) -> dict[str, object]:
            fragment = self._read_spectrum_fragment(index)
            header = fragment.split(b"<binaryDataArrayList", 1)[0]
            params = _cv_params_bytes(header)

            time_value = 0.0
            time_unit_accession = None
            time_unit_name = None
            for param in params:
                if param.get("accession") == "MS:1000016":
                    try:
                        time_value = float(param.get("value", "0"))
                    except ValueError:
                        time_value = 0.0
                    time_unit_accession = param.get("unitAccession")
                    time_unit_name = param.get("unitName")
                    break

            def number(accession: str) -> float | None:
                value = _param_value(params, accession)
                try:
                    return float(value) if value is not None else None
                except ValueError:
                    return None

            try:
                level = int(float(_param_value(params, "MS:1000511", "1") or "1"))
            except ValueError:
                level = 1

            mode = "profile" if _has_param(params, "MS:1000128") else (
                "centroid" if _has_param(params, "MS:1000127") else "unknown"
            )
            return {
                "time_min": _time_to_minutes(time_value, time_unit_accession, time_unit_name),
                "tic": number("MS:1000285"),
                "ms_level": level,
                "mode": mode,
                "observed_low": number("MS:1000528"),
                "observed_high": number("MS:1000527"),
                "scan_window_low": number("MS:1000501"),
                "scan_window_high": number("MS:1000500"),
            }

        def _load_tic_or_headers(self) -> None:
            loaded = False
            tic_offset = None
            for identifier, offset in self._chromatogram_offsets.items():
                if identifier.upper() == "TIC" or "total ion" in identifier.lower():
                    tic_offset = offset
                    break
            if tic_offset is not None and self._index_list_offset is not None:
                try:
                    fragment = self._read_range(tic_offset, self._index_list_offset)
                    element = _parse_xml_fragment(fragment, "chromatogram")
                    arrays: dict[str, np.ndarray] = {}
                    for child in element.iter():
                        if _local_name(child.tag) == "binaryDataArray":
                            kind, values = _decode_binary_array(child, self.reference_groups)
                            if kind:
                                arrays[kind] = values
                    times = arrays.get("time")
                    tic = arrays.get("intensity")
                    if times is not None and tic is not None and len(times) == self.n_scans:
                        params = _element_params(element, self.reference_groups)
                        unit_accession = None
                        unit_name = None
                        for child in element.iter():
                            if _local_name(child.tag) != "binaryDataArray":
                                continue
                            child_params = _element_params(child, self.reference_groups)
                            if _has_param(child_params, "MS:1000595"):
                                time_param = next(
                                    (p for p in child_params if p.get("accession") == "MS:1000595"),
                                    {},
                                )
                                unit_accession = time_param.get("unitAccession")
                                unit_name = time_param.get("unitName")
                                break
                        factor_times = np.asarray(times, dtype=float)
                        if unit_accession == "UO:0000010" or (unit_name or "").lower().startswith("second"):
                            factor_times = factor_times / 60.0
                        elif unit_accession == "UO:0000032" or (unit_name or "").lower().startswith("hour"):
                            factor_times = factor_times * 60.0
                        self.times_min = factor_times
                        self.tic = np.asarray(tic, dtype=float)
                        loaded = True
                except Exception:
                    loaded = False

            if loaded:
                return

            times = np.empty(self.n_scans, dtype=float)
            tic = np.empty(self.n_scans, dtype=float)
            for index in range(self.n_scans):
                metadata = self._read_header_metadata(index)
                times[index] = float(metadata.get("time_min", index))
                tic_value = metadata.get("tic")
                if tic_value is None:
                    record = self.read_spectrum(index)
                    tic[index] = float(np.nansum(record.intensity))
                else:
                    tic[index] = float(tic_value)
            self.times_min = times
            self.tic = tic

        def read_spectrum(self, index: int) -> SpectrumRecord:
            index = int(np.clip(index, 0, self.n_scans - 1))
            fragment = self._read_spectrum_fragment(index)
            element = _parse_xml_fragment(fragment, "spectrum")
            params = _element_params(element, self.reference_groups)

            try:
                level = int(float(_param_value(params, "MS:1000511", "1") or "1"))
            except ValueError:
                level = 1
            mode = "profile" if _has_param(params, "MS:1000128") else (
                "centroid" if _has_param(params, "MS:1000127") else "unknown"
            )

            time_min = float(self.times_min[index]) if self.times_min.size else float(index)
            for param in params:
                if param.get("accession") == "MS:1000016":
                    try:
                        value = float(param.get("value", "0"))
                        time_min = _time_to_minutes(
                            value, param.get("unitAccession"), param.get("unitName")
                        )
                    except ValueError:
                        pass
                    break

            arrays: dict[str, np.ndarray] = {}
            for child in element.iter():
                if _local_name(child.tag) != "binaryDataArray":
                    continue
                kind, values = _decode_binary_array(child, self.reference_groups)
                if kind:
                    arrays[kind] = values

            mz = np.asarray(arrays.get("mz", np.array([], dtype=float)), dtype=float)
            intensity = np.asarray(
                arrays.get("intensity", np.array([], dtype=float)), dtype=float
            )
            if mz.size != intensity.size:
                size = min(mz.size, intensity.size)
                mz = mz[:size]
                intensity = intensity[:size]

            tic_value = _param_value(params, "MS:1000285")
            try:
                tic = float(tic_value) if tic_value is not None else float(np.nansum(intensity))
            except ValueError:
                tic = float(np.nansum(intensity))

            identifier = element.attrib.get("id", self.spectrum_ids[index])
            return SpectrumRecord(
                index=index,
                scan_number=_scan_number(identifier, int(self.scan_numbers[index])),
                spectrum_id=identifier,
                time_min=time_min,
                ms_level=level,
                mode=mode,
                tic=tic,
                mz=mz,
                intensity=intensity,
            )

        def nearest_scan_index(self, time_min: float) -> int:
            return int(np.nanargmin(np.abs(self.times_min - float(time_min))))

        def indices_in_time_range(self, low: float, high: float) -> np.ndarray:
            low, high = sorted((float(low), float(high)))
            return np.flatnonzero((self.times_min >= low) & (self.times_min <= high))

        def iter_spectra(
            self, indices: Iterable[int], *, ms_level: int | None = None
        ) -> Iterator[SpectrumRecord]:
            for raw_index in indices:
                index = int(raw_index)
                if not 0 <= index < self.n_scans:
                    continue
                record = self.read_spectrum(index)
                if ms_level is None or record.ms_level == int(ms_level):
                    yield record

        def estimate_sampling_resolution(
            self,
            indices: Iterable[int],
            *,
            mz_min: float,
            mz_max: float,
            ms_level: int | None = 1,
            sample_count: int = 32,
        ) -> float:
            """Estimate native sampling resolution from representative spectra.

            The densest representative spectrum is used, following UniDec's
            practical strategy of favoring the scan with the most profile points.
            """
            normalized = np.asarray(
                sorted({int(i) for i in indices if 0 <= int(i) < self.n_scans}),
                dtype=np.int64,
            )
            if normalized.size == 0:
                raise ValueError("No scans are available for sampling-resolution estimation")
            take = min(int(sample_count), int(normalized.size))
            positions = np.linspace(0, normalized.size - 1, take, dtype=np.int64)
            candidates = normalized[np.unique(positions)]
            best_count = -1
            best_spacing: float | None = None
            for record in self.iter_spectra(candidates, ms_level=ms_level):
                mz, intensity = _prepare_spectrum_arrays(record.mz, record.intensity)
                keep = (mz >= float(mz_min)) & (mz <= float(mz_max))
                mz = mz[keep]
                if mz.size < 3:
                    continue
                spacing = _typical_relative_spacing(mz)
                if spacing is not None and mz.size > best_count:
                    best_count = int(mz.size)
                    best_spacing = spacing
            if best_spacing is None:
                raise ValueError("Could not estimate the native m/z sampling resolution")
            return float(1.0 / best_spacing)

        def sum_spectra(
            self,
            indices: Iterable[int],
            *,
            mz_min: float,
            mz_max: float,
            ms_level: int | None = 1,
            grid_mode: str = "auto_resolution",
            oversampling: float = 4.0,
            fixed_step: float = 0.01,
            profile_method: str = "area_corrected",
            gap_factor: float = 8.0,
            max_grid_points: int = 2_000_000,
            progress: Callable[[int, int], None] | None = None,
        ) -> SumResult:
            normalized = np.asarray(
                sorted({int(i) for i in indices if 0 <= int(i) < self.n_scans}),
                dtype=np.int64,
            )
            if normalized.size == 0:
                return SumResult(
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                    0,
                    0,
                    None,
                    None,
                    ms_level,
                    profile_method,
                    grid_mode,
                    None,
                    None,
                )

            low = float(mz_min)
            high = float(mz_max)
            if not (np.isfinite(low) and np.isfinite(high) and high > low):
                raise ValueError("The m/z limits are invalid")
            grid_mode = str(grid_mode).lower().strip()
            profile_method = str(profile_method).lower().strip()
            if grid_mode not in {"auto_resolution", "fixed_da"}:
                raise ValueError("grid_mode must be 'auto_resolution' or 'fixed_da'")
            if profile_method not in {"area_corrected", "unidec_points", "legacy_linear"}:
                raise ValueError(
                    "profile_method must be 'area_corrected', 'unidec_points', or 'legacy_linear'"
                )
            if not np.isfinite(gap_factor) or gap_factor <= 1:
                raise ValueError("The profile gap factor must be greater than one")

            sampling_resolution: float | None = None
            used_oversampling: float | None = None
            if grid_mode == "auto_resolution":
                sampling_resolution = self.estimate_sampling_resolution(
                    normalized,
                    mz_min=low,
                    mz_max=high,
                    ms_level=ms_level,
                )
                used_oversampling = float(oversampling)
                edges = _constant_resolution_edges(
                    low,
                    high,
                    sampling_resolution,
                    used_oversampling,
                    max_grid_points,
                )
            else:
                edges = _fixed_da_edges(low, high, float(fixed_step), max_grid_points)

            widths = np.diff(edges)
            centers = np.sqrt(edges[:-1] * edges[1:]) if np.all(edges > 0) else 0.5 * (
                edges[:-1] + edges[1:]
            )
            area_accumulator = np.zeros(widths.size, dtype=np.float64)
            density_accumulator = np.zeros(widths.size, dtype=np.float64)
            included = 0
            total = int(normalized.size)

            for position, record in enumerate(
                self.iter_spectra(normalized, ms_level=ms_level), start=1
            ):
                if record.mode == "centroid":
                    _deposit_points(area_accumulator, edges, record.mz, record.intensity)
                elif profile_method == "area_corrected":
                    _deposit_profile_area(
                        area_accumulator,
                        edges,
                        record.mz,
                        record.intensity,
                        gap_factor=float(gap_factor),
                    )
                elif profile_method == "unidec_points":
                    _deposit_points(area_accumulator, edges, record.mz, record.intensity)
                else:
                    _deposit_legacy_interpolation(
                        density_accumulator,
                        centers,
                        record.mz,
                        record.intensity,
                        gap_factor=float(gap_factor),
                    )
                included += 1
                if progress is not None:
                    progress(position, total)

            if profile_method == "legacy_linear":
                intensity = density_accumulator
                bin_area = density_accumulator * widths
            elif profile_method == "unidec_points":
                # UniDec-like point sums are kept as binned intensity values. Their
                # area interpretation depends on the source sampling density.
                intensity = area_accumulator.copy()
                bin_area = area_accumulator.copy()
            else:
                bin_area = area_accumulator
                intensity = np.divide(
                    bin_area,
                    widths,
                    out=np.zeros_like(bin_area),
                    where=widths > 0,
                )

            return SumResult(
                mz=centers,
                intensity=intensity,
                bin_edges=edges,
                bin_area=bin_area,
                requested_count=total,
                included_count=included,
                first_index=int(normalized[0]),
                last_index=int(normalized[-1]),
                ms_level=ms_level,
                method=profile_method,
                grid_mode=grid_mode,
                sampling_resolution=sampling_resolution,
                oversampling=used_oversampling,
            )


    def reduce_xy_minmax(
        x: np.ndarray,
        y: np.ndarray,
        *,
        max_points: int = 30_000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Peak-preserving min/max envelope reduction for interactive plotting."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        if n <= max_points or max_points < 4:
            return x.copy(), y.copy(), np.arange(n, dtype=np.int64)

        bucket_count = max(1, max_points // 2)
        edges = np.linspace(0, n, bucket_count + 1, dtype=np.int64)
        chosen: list[int] = []
        for start, stop in zip(edges[:-1], edges[1:]):
            if stop <= start:
                continue
            segment = y[start:stop]
            finite = np.isfinite(segment)
            if not np.any(finite):
                chosen.append(start)
                continue
            finite_indices = np.flatnonzero(finite)
            local_values = segment[finite]
            local_min = int(finite_indices[int(np.argmin(local_values))]) + start
            local_max = int(finite_indices[int(np.argmax(local_values))]) + start
            if local_min <= local_max:
                chosen.extend((local_min, local_max))
            else:
                chosen.extend((local_max, local_min))
        indices = np.unique(np.asarray(chosen, dtype=np.int64))
        return x[indices], y[indices], indices

    return (
        IndexedMzMLSource,
        MzMLIndexedReaderError,
        Path,
        np,
        re,
        reduce_xy_minmax,
    )


@app.cell
def _():
    import json
    import sys
    import tempfile

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go

    # In a static WASM/Pyodide export there is no server filesystem, so only the
    # in-browser uploader is usable. Under ``marimo run``/``marimo edit`` the
    # path-based file browser is available and preferred for large files.
    IS_WASM = sys.platform == "emscripten"
    return IS_WASM, go, json, mo, pd, tempfile


@app.cell
def _(mo):
    # --- Palette (distinct colors for the two trace families) -----------------
    COLOR_TIC = "#2563EB"   # blue  -> total ion chromatogram
    COLOR_MS = "#EA580C"    # orange-> mass spectral trace
    COLOR_WIN = "#7C3AED"   # violet-> extraction-window shading
    FONT_STACK = (
        "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    )
    PLOT_TEMPLATE = "plotly_white"

    app_css = mo.Html(
        """
        <style>
        :root {
            --font-stack: system-ui, -apple-system, "Segoe UI", Roboto,
                          Helvetica, Arial, sans-serif;
            --card-border: #e2e8f0;
            --card-bg: #ffffff;
            --muted: #64748b;
            --accent: #2563eb;
        }
        /* marimo's own type tokens default to Lora/PT Sans (serif headings).
           Reset them at the root so every heading and prose block is sans. */
        :root, body, .marimo, .Marimo {
            --marimo-heading-font: system-ui, -apple-system, "Segoe UI", Roboto,
                                   Helvetica, Arial, sans-serif;
            --marimo-text-font: system-ui, -apple-system, "Segoe UI", Roboto,
                                Helvetica, Arial, sans-serif;
            --heading-font: system-ui, -apple-system, "Segoe UI", Roboto,
                            Helvetica, Arial, sans-serif;
            --text-font: system-ui, -apple-system, "Segoe UI", Roboto,
                         Helvetica, Arial, sans-serif;
        }
        html, body, .marimo, .Marimo, .markdown, .prose,
        .markdown h1, .markdown h2, .markdown h3,
        .markdown h4, .markdown h5, .markdown h6,
        h1, h2, h3, h4, h5, h6, p, span, div, label,
        button, input, select, textarea, table, th, td, summary {
            font-family: system-ui, -apple-system, "Segoe UI", Roboto,
                         Helvetica, Arial, sans-serif !important;
        }
        .card-sub {
            color: var(--muted); font-size: 0.82rem; line-height: 1.35;
            white-space: normal; overflow-wrap: anywhere;
        }
        .plot-note { color: var(--muted); font-size: 0.8rem; margin-top: 4px; }
        /* Titled divider: a labelled horizontal rule that groups the settings
           without hiding them behind a click. */
        .section-rule {
            display: flex; align-items: center; gap: 10px;
            margin: 6px 0 2px; color: var(--muted);
            font-size: 0.78rem; font-weight: 700;
            letter-spacing: 0.06em; text-transform: uppercase;
            white-space: nowrap;
        }
        .section-rule::after {
            content: ""; flex: 1 1 auto; height: 1px;
            background: var(--card-border);
        }
        /* Flex children default to min-width:auto, which lets the wide data
           editor push the right column past the viewport edge. marimo's stack
           wrappers are inline-styled divs, so force them to be shrinkable;
           the cards themselves carry overflow-x:auto and scroll internally. */
        .peakachu-col { min-width: 0; max-width: 100%; }
        [style*="display: flex"] > * { min-width: 0 !important; }
        .peakachu-nowrap, .peakachu-nowrap * { white-space: nowrap; }
        button { white-space: nowrap; }
        .mzml-help {
            cursor: help; font-weight: 700; border: 1px solid currentColor;
            border-radius: 50%; display: inline-flex; width: 1.1rem; height: 1.1rem;
            align-items: center; justify-content: center; opacity: 0.6;
            font-size: 0.72rem;
        }

        /* --- Animated "aurora" title -------------------------------------
           Blurred colour blobs drift behind the word and are composited into
           the glyphs with mix-blend-mode. The blending needs white text on a
           dark ground, so the banner carries its own dark background. */
        .peakachu-banner {
            --bg: #0b0f19;
            --clr-1: #00c2ff;
            --clr-2: #33ff8c;
            --clr-3: #ffc640;
            --clr-4: #e54cff;
            --blur: 1rem;
            background: var(--bg);
            border-radius: 14px;
            padding: 18px 22px 20px;
            margin-bottom: 4px;
            overflow: hidden;
        }
        .peakachu-title {
            position: relative;
            overflow: hidden;
            display: inline-block;
            margin: 0;
            padding: 0 0.06em;
            background: var(--bg);
            color: #fff;
            font-size: clamp(2.1rem, 5vw, 3.4rem);
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.12;
        }
        .peakachu-aurora {
            position: absolute;
            inset: 0;
            z-index: 2;
            mix-blend-mode: darken;
            pointer-events: none;
        }
        .peakachu-aurora__item {
            position: absolute;
            width: 60%;
            height: 100%;
            background-color: var(--clr-1);
            border-radius: 37% 29% 27% 27% / 28% 25% 41% 37%;
            filter: blur(var(--blur));
            mix-blend-mode: overlay;
            animation: aurora-border 6s ease-in-out infinite;
        }
        .peakachu-aurora__item:nth-of-type(1) {
            top: -50%; left: 0;
            background-color: var(--clr-1);
            animation: aurora-border 6s ease-in-out infinite,
                       aurora-1 12s ease-in-out infinite alternate;
        }
        .peakachu-aurora__item:nth-of-type(2) {
            top: 0; right: 0;
            background-color: var(--clr-2);
            animation: aurora-border 6s ease-in-out infinite,
                       aurora-2 12s ease-in-out infinite alternate;
        }
        .peakachu-aurora__item:nth-of-type(3) {
            bottom: 0; left: 0;
            background-color: var(--clr-3);
            animation: aurora-border 6s ease-in-out infinite,
                       aurora-3 8s ease-in-out infinite alternate;
        }
        .peakachu-aurora__item:nth-of-type(4) {
            bottom: -50%; right: 0;
            background-color: var(--clr-4);
            animation: aurora-border 6s ease-in-out infinite,
                       aurora-4 24s ease-in-out infinite alternate;
        }
        .peakachu-sub {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 6px;
            letter-spacing: 0.01em;
        }
        @keyframes aurora-1 {
            0%   { top: 0;    right: 0; }
            50%  { top: 100%; right: 75%; }
            75%  { top: 100%; right: 25%; }
            100% { top: 0;    right: 0; }
        }
        @keyframes aurora-2 {
            0%   { top: -50%; left: 0; }
            60%  { top: 100%; left: 75%; }
            85%  { top: 100%; left: 25%; }
            100% { top: -50%; left: 0; }
        }
        @keyframes aurora-3 {
            0%   { bottom: 0;    left: 0; }
            40%  { bottom: 100%; left: 75%; }
            65%  { bottom: 40%;  left: 50%; }
            100% { bottom: 0;    left: 0; }
        }
        @keyframes aurora-4 {
            0%   { bottom: -50%; right: 0; }
            50%  { bottom: 0;    right: 40%; }
            90%  { bottom: 50%;  right: 25%; }
            100% { bottom: -50%; right: 0; }
        }
        @keyframes aurora-border {
            0%   { border-radius: 37% 29% 27% 27% / 28% 25% 41% 37%; }
            25%  { border-radius: 47% 29% 39% 49% / 61% 19% 66% 26%; }
            50%  { border-radius: 57% 23% 47% 72% / 63% 17% 66% 33%; }
            75%  { border-radius: 28% 49% 29% 100% / 93% 20% 64% 25%; }
            100% { border-radius: 37% 29% 27% 27% / 28% 25% 41% 37%; }
        }
        @media (prefers-reduced-motion: reduce) {
            .peakachu-aurora__item { animation: none !important; }
        }
        </style>
        """
    )

    def title_banner(name, tagline):
        """Dark banner whose word is filled by drifting aurora colour blobs."""
        return mo.Html(
            f"""
            <div class="peakachu-banner">
              <h1 class="peakachu-title">{name}
                <div class="peakachu-aurora">
                  <div class="peakachu-aurora__item"></div>
                  <div class="peakachu-aurora__item"></div>
                  <div class="peakachu-aurora__item"></div>
                  <div class="peakachu-aurora__item"></div>
                </div>
              </h1>
              <div class="peakachu-sub">{tagline}</div>
            </div>
            """
        )

    def card(*content, title=None, subtitle=None):
        """Wrap content in a light, consistent card container."""
        items = []
        if title is not None:
            items.append(mo.md(f"#### {title}"))
        if subtitle is not None:
            items.append(mo.Html(f"<div class='card-sub'>{subtitle}</div>"))
        items.extend(content)
        return mo.vstack(items, gap=0.5).style(
            {
                "border": "1px solid var(--card-border)",
                "border-radius": "14px",
                "padding": "14px 16px",
                "background": "var(--card-bg)",
                "box-shadow": "0 1px 2px rgba(15,23,42,0.05)",
                "width": "100%",
                "min-width": "0",
                "max-width": "100%",
                "overflow-x": "auto",
                "box-sizing": "border-box",
            }
        )

    def help_tip(text):
        import html as _html

        return mo.Html(
            f"<span class='mzml-help' title=\"{_html.escape(text, quote=True)}\">?</span>"
        )

    return (
        COLOR_MS,
        COLOR_TIC,
        COLOR_WIN,
        FONT_STACK,
        PLOT_TEMPLATE,
        app_css,
        card,
        help_tip,
        title_banner,
    )


@app.cell
def _(IndexedMzMLSource, Path, np, tempfile):
    # --- Session utilities (shared by single-file and batch workflows) --------
    TMPDIR = Path(tempfile.mkdtemp(prefix="mzml_viewer_"))
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    def write_temp_mzml(name, contents):
        """Persist uploaded bytes to the temp filesystem for byte-range access."""
        safe = Path(str(name)).name or "upload.mzML"
        if not safe.lower().endswith((".mzml",)):
            safe = safe + ".mzML"
        target = TMPDIR / safe
        target.write_bytes(bytes(contents))
        return target

    def load_uploaded_source(upload):
        """Write one FileUploadResults to disk and open it as a source."""
        path = write_temp_mzml(upload.name, upload.contents)
        return IndexedMzMLSource(path)

    def load_source_from_path(path):
        """Open a file already on disk by path (no copy, no whole-file read).

        The reader seeks by byte offset, so this streams from the original file
        and sidesteps the browser-upload size limit. Server mode only.
        """
        return IndexedMzMLSource(Path(str(path)))

    def active_windows(rows):
        """Return validated (label, low, high) tuples for windows marked Use."""
        result = []
        for i, row in enumerate(rows or []):
            if not bool(row.get("Use", True)):
                continue
            try:
                low = float(row.get("m/z minimum"))
                high = float(row.get("m/z maximum"))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(low) and np.isfinite(high)):
                continue
            low, high = sorted((low, high))
            label = str(row.get("Label") or f"Window {i + 1}")
            result.append((label, low, high))
        return result

    def integrate_window(mz, intensity, low, high):
        """Trapezoidal integral of intensity within [low, high]."""
        mask = (mz >= low) & (mz <= high)
        if not np.any(mask):
            return 0.0
        x = mz[mask]
        y = intensity[mask]
        if x.size > 1:
            order = np.argsort(x)
            return float(_trapz(y[order], x[order]))
        return float(np.nansum(y))

    def stream_integrate(source, indices, windows, ms_level):
        """Stream spectra once, integrating every window plus a grand total."""
        totals = {label: 0.0 for label, _lo, _hi in windows}
        grand = 0.0
        level = None if not ms_level else int(ms_level)
        for record in source.iter_spectra(indices, ms_level=level):
            mz = np.asarray(record.mz, dtype=float)
            inten = np.asarray(record.intensity, dtype=float)
            if mz.size == 0:
                continue
            if mz.size > 1 and not np.all(np.diff(mz) >= 0):
                order = np.argsort(mz)
                mz = mz[order]
                inten = inten[order]
            grand += float(_trapz(inten, mz)) if mz.size > 1 else float(np.nansum(inten))
            for label, low, high in windows:
                totals[label] += integrate_window(mz, inten, low, high)
        return totals, grand

    def estimate_native_resolution(source, mz_min, mz_max, ms_level=None):
        """Native sampling resolution the reader would use for this grid.

        Delegates to the reader's own public estimator rather than
        re-deriving it. That matters: the reader keys off the *densest*
        representative spectrum, not a median, so any home-grown median-of-N
        estimate lands ~1.5% low -- which would let the budget report "fits"
        for a grid that then blows the cap. Same call, same answer, no drift.
        """
        if source is None or source.n_scans == 0:
            return None
        try:
            return float(
                source.estimate_sampling_resolution(
                    np.arange(source.n_scans, dtype=np.int64),
                    mz_min=float(mz_min),
                    mz_max=float(mz_max),
                    ms_level=ms_level,
                )
            )
        except Exception:  # noqa: BLE001 - budget display must never break the app
            return None

    def predict_bins(grid_mode, low, high, resolution, oversampling, fixed_step):
        """Bin count the reader would produce, without building the grid."""
        if not (high > low):
            return None
        if str(grid_mode) == "auto_resolution":
            if not resolution or low <= 0 or oversampling <= 0:
                return None
            dlog = np.log1p(1.0 / (float(resolution) * float(oversampling)))
            if not np.isfinite(dlog) or dlog <= 0:
                return None
            return int(np.ceil(np.log(high / low) / dlog))
        if float(fixed_step) <= 0:
            return None
        return int(np.ceil((high - low) / float(fixed_step)))

    def fit_oversampling(low, high, resolution, cap, preferred=4.0):
        """Largest oversampling <= preferred whose grid fits inside `cap` bins.

        Bins scale linearly with oversampling, so solve directly. 2% headroom
        and a floor to 1 dp keep the answer from landing exactly on the cap.
        """
        bins = predict_bins("auto_resolution", low, high, resolution, preferred, 1.0)
        if bins is None or bins <= cap:
            return float(preferred)
        scaled = int(preferred * cap / bins * 0.98 * 10) / 10.0
        return float(max(1.0, scaled))

    return (
        TMPDIR,
        active_windows,
        estimate_native_resolution,
        fit_oversampling,
        integrate_window,
        load_source_from_path,
        load_uploaded_source,
        predict_bins,
        stream_integrate,
        write_temp_mzml,
    )


@app.cell
def _(mo):
    # --- Reactive session state ----------------------------------------------
    get_source, set_source = mo.state(None)
    get_load_error, set_load_error = mo.state(None)
    get_view_request, set_view_request = mo.state(
        {"kind": "scan", "index": 0, "source": "initial"}
    )
    get_mass_windows, set_mass_windows = mo.state(
        [
            {
                "Use": True,
                "Label": "Window 1",
                "m/z minimum": 500.0,
                "m/z maximum": 501.0,
            }
        ]
    )
    # Windows persist across files. Once the user touches them we never
    # auto-overwrite; before that we fit the first window to the loaded range.
    get_windows_touched, set_windows_touched = mo.state(False)
    get_batch_result, set_batch_result = mo.state(None)
    return (
        get_batch_result,
        get_load_error,
        get_mass_windows,
        get_source,
        get_view_request,
        get_windows_touched,
        set_batch_result,
        set_load_error,
        set_mass_windows,
        set_source,
        set_view_request,
        set_windows_touched,
    )


@app.cell
def _(
    IS_WASM,
    get_mass_windows,
    get_windows_touched,
    load_source_from_path,
    load_uploaded_source,
    mo,
    set_load_error,
    set_mass_windows,
    set_source,
    set_view_request,
):
    # --- Single-file loader ---------------------------------------------------
    # Two entry points:
    #   * mo.ui.file  -> browser upload, works everywhere incl. WASM, 100 MB cap.
    #   * mo.ui.file_browser -> picks a path on disk; the byte-range reader seeks
    #     into it without loading the whole file, so there is no size limit.
    #     Only meaningful when a real filesystem exists (server mode).
    def _fit_first_window(source):
        low, high = source.default_mz_min, source.default_mz_max
        center = 0.5 * (low + high)
        half = max(0.5, (high - low) * 0.002)
        set_mass_windows(
            [
                {
                    "Use": True,
                    "Label": "Window 1",
                    "m/z minimum": round(max(low, center - half), 4),
                    "m/z maximum": round(min(high, center + half), 4),
                }
            ]
        )

    def _adopt(source):
        set_source(source)
        set_load_error(None)
        set_view_request({"kind": "scan", "index": 0, "source": "new_file"})
        # Persist existing window selection; only fit on the very first load.
        if not get_windows_touched() and get_mass_windows():
            _fit_first_window(source)

    def _on_upload(files):
        if not files:
            return
        try:
            source = load_uploaded_source(files[0])
        except Exception as exc:  # noqa: BLE001 - surface reader errors to the UI
            set_source(None)
            set_load_error(str(exc))
            return
        _adopt(source)

    def _on_browse(selection):
        if not selection:
            return
        item = selection[0]
        path = getattr(item, "path", None) or getattr(item, "id", None)
        if path is None:
            return
        try:
            source = load_source_from_path(path)
        except Exception as exc:  # noqa: BLE001
            set_source(None)
            set_load_error(str(exc))
            return
        _adopt(source)

    def _clear(_value):
        set_source(None)
        set_load_error(None)
        set_view_request({"kind": "scan", "index": 0, "source": "clear"})

    upload_widget = mo.ui.file(
        filetypes=[".mzML", ".mzml"],
        kind="area",
        label="Drop or choose an uncompressed .mzML file (up to 100 MB)",
        on_change=_on_upload,
    )
    path_browser = None
    if not IS_WASM:
        path_browser = mo.ui.file_browser(
            filetypes=[".mzML", ".mzml"],
            selection_mode="file",
            multiple=False,
            label="",
            on_change=_on_browse,
        )
    clear_button = mo.ui.button(
        label="Clear loaded file",
        tooltip="Release the current index and displayed spectrum. Windows are kept.",
        on_click=_clear,
    )
    return clear_button, path_browser, upload_widget


@app.cell
def _(card, get_load_error, get_source, mo):
    # --- Consolidated load-status card (replaces two redundant callouts) ------
    _source = get_source()
    _error = get_load_error()
    if _error:
        _status = mo.callout(
            mo.md(f"**Could not open the file**\n\n`{_error}`"), kind="danger"
        )
    elif _source is None:
        _status = mo.callout(
            "Upload a file to read its spectrum index and total ion chromatogram.",
            kind="info",
        )
    else:
        _index_type = (
            "embedded indexedmzML offsets"
            if _source.is_indexed
            else "locally rebuilt offset index"
        )
        _status = mo.md(
            f"**Loaded** `{_source.name}`  \n"
            f"**Spectra** {_source.n_scans:,}"
            f" &nbsp;·&nbsp; **RT** {_source.time_min:.4f}–{_source.time_max:.4f} min  \n"
            f"**First scan** MS{_source.first_ms_level}, {_source.first_mode}"
            f" &nbsp;·&nbsp; **Default m/z** {_source.default_mz_min:.2f}–{_source.default_mz_max:.2f}  \n"
            f"**Index** {_index_type}"
            f" &nbsp;·&nbsp; **Resident** {_source.compact_memory_bytes / 1024**2:.2f} MB"
        )
    load_status_card = card(_status, title="File status")
    return (load_status_card,)


@app.cell
def _(IS_WASM, card, mo):
    _limit_note = (
        """
**Large files.** Browser upload is capped at **100 MB** (a marimo limit), and
oversized uploads can fail silently. This static WASM build has no server
filesystem, so upload is the only option here — for bigger acquisitions, run the
notebook with `marimo run` and use the on-disk file browser, which streams by
byte range with no size limit.
        """
        if IS_WASM
        else """
**Large files.** Browser upload is capped at **100 MB** (a marimo limit) and
reads the whole file into memory. For bigger acquisitions use the **file
browser** above: it hands the reader a path and the spectra are read by byte
range straight from disk, so there is no practical size limit.
        """
    )
    requirements_card = card(
        mo.md(
            """
Use an uncompressed `.mzML`. Files with embedded `indexedmzML` offsets load
fastest; a plain mzML is accepted by rebuilding the offset table on first read.
`.mzML.gz` is rejected because gzip has no efficient random access.
"""
            + _limit_note
        ),
        title="File requirements",
    )
    return (requirements_card,)


@app.cell
def _(
    card,
    clear_button,
    load_status_card,
    mo,
    path_browser,
    requirements_card,
    upload_widget,
):
    if path_browser is not None:
        _load_body = [
            mo.md("**Upload** (≤ 100 MB) — works in any deployment:"),
            upload_widget,
            mo.md(
                "**Or pick a file on disk** — no size limit, read by byte range "
                "(available when running with `marimo run`):"
            ),
            path_browser.style({"max-height": "260px", "overflow": "auto"}),
            mo.hstack([clear_button], justify="start"),
        ]
    else:
        _load_body = [
            upload_widget,
            mo.hstack([clear_button], justify="start"),
        ]
    load_panel = mo.vstack(
        [
            card(
                *_load_body,
                title="Load an mzML file",
                subtitle="Runs locally with <code>marimo run</code> or as a static WASM export.",
            ),
            mo.hstack(
                [load_status_card, requirements_card],
                widths=[0.55, 0.45],
                gap=0.8,
                align="stretch",
            ),
        ],
        gap=0.8,
    )
    return (load_panel,)


@app.cell
def _(
    card,
    estimate_native_resolution,
    fit_oversampling,
    get_source,
    help_tip,
    mo,
    set_view_request,
):
    # --- Viewer controls (advanced summation tucked into an accordion) --------
    _source = get_source()
    _disabled = _source is None
    _dmin = _source.default_mz_min if _source is not None else 100.0
    _dmax = _source.default_mz_max if _source is not None else 2000.0
    _dlevel = _source.first_ms_level if _source is not None else 1

    # --- Grid budget estimator -------------------------------------------
    # Run once per loaded file: measure the native sampling resolution, then
    # preset Oversampling to the largest value (<= 4) whose full-range sum
    # fits inside the default bin budget. This is what stops a "Sum all" from
    # dying on the bin cap the first time it is pressed on wide, high-R data.
    _BIN_BUDGET = 2_000_000
    # ~1.5 s on a large file (it decodes up to 32 spectra to find the densest),
    # so this runs once per loaded file and the result is reused everywhere.
    if _source is not None:
        with mo.status.spinner(title="Estimating native sampling resolution"):
            native_resolution = estimate_native_resolution(_source, _dmin, _dmax, _dlevel)
    else:
        native_resolution = None
    _ov_default = (
        fit_oversampling(_dmin, _dmax, native_resolution, _BIN_BUDGET, preferred=4.0)
        if _source is not None
        else 4.0
    )

    interaction_mode = mo.ui.dropdown(
        options={"Zoom": "zoom", "Select a time window": "select"},
        value="Zoom",
        label="TIC drag",
        disabled=_disabled,
    )
    spectrum_mode = mo.ui.dropdown(
        options={"Zoom": "zoom", "Select an m/z window": "select"},
        value="Zoom",
        label="Spectrum drag",
        disabled=_disabled,
    )
    ms_level_control = mo.ui.dropdown(
        options={"MS1": 1, "MS2": 2, "MS3": 3, "All levels": 0},
        value=f"MS{_dlevel}" if _dlevel in (1, 2, 3) else "MS1",
        label="MS level",
        disabled=_disabled,
    )
    sum_all_button = mo.ui.button(
        label="Sum all spectra",
        kind="success",
        disabled=_disabled,
        tooltip="Stream every selected-level spectrum into one summed spectrum.",
        on_click=lambda _v: set_view_request({"kind": "all", "source": "sum_all_button"}),
    )
    mz_min_control = mo.ui.number(
        value=float(_dmin), step=0.1, label="Sum m/z min", disabled=_disabled, full_width=True
    )
    mz_max_control = mo.ui.number(
        value=float(_dmax), step=0.1, label="Sum m/z max", disabled=_disabled, full_width=True
    )
    grid_mode_control = mo.ui.dropdown(
        options={
            "Automatic constant-resolution grid": "auto_resolution",
            "Fixed-Da grid": "fixed_da",
        },
        value="Automatic constant-resolution grid",
        label="Grid mode",
        disabled=_disabled,
    )
    oversampling_control = mo.ui.number(
        start=1.0, stop=16.0, value=_ov_default, step=0.1,
        label="Oversampling", disabled=_disabled, full_width=True,
    )
    fixed_step_control = mo.ui.number(
        start=0.0001, value=0.01, step=0.001,
        label="Fixed bin (Da)", disabled=_disabled, full_width=True,
    )
    profile_method_control = mo.ui.dropdown(
        options={
            "Area-corrected profile integration": "area_corrected",
            "UniDec-style point integration": "unidec_points",
            "Legacy segmented interpolation": "legacy_linear",
        },
        value="Area-corrected profile integration",
        label="Integration",
        disabled=_disabled,
    )
    gap_factor_control = mo.ui.number(
        start=2.0, stop=100.0, value=8.0, step=1.0,
        label="Gap threshold", disabled=_disabled, full_width=True,
    )
    # The reader caps the common grid to guard memory: every bin costs ~24 B
    # across the m/z, intensity and area arrays. 2 M bins is a sane browser
    # default; locally there is room to raise it for wide, high-resolution runs.
    max_bins_control = mo.ui.number(
        start=100_000, stop=20_000_000, value=2_000_000, step=100_000,
        label="Max grid bins", disabled=_disabled, full_width=True,
    )

    # An accordion always renders collapsed (marimo has no default-open flag),
    # so these settings were effectively hidden behind an unlabelled disclosure.
    # Show them outright under a titled rule instead -- the card sits below the
    # plots, so there is room and nothing important is pushed off screen.
    _settings = mo.vstack(
        [
            mo.Html(
                "<div style='display:flex;align-items:center;gap:10px;margin:10px 0 2px;color:#64748b;font-size:0.72rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;white-space:nowrap;'>"
                "<span>Summation grid</span>"
                "<span style='flex:1 1 auto;height:1px;background:#e2e8f0;'></span></div>"
            ),
            mo.hstack(
                [mz_min_control, help_tip("Lower edge of the common summation grid."),
                 mz_max_control, help_tip("Upper edge of the common summation grid.")],
                widths=[0.44, 0.06, 0.44, 0.06], gap=0.2, align="center",
            ),
            mo.hstack(
                [grid_mode_control,
                 help_tip("Automatic grid tracks native sampling; bin width grows with m/z."),
                 profile_method_control,
                 help_tip("Area-corrected integration is the quantitative default.")],
                widths=[0.44, 0.06, 0.44, 0.06], gap=0.2, align="center",
            ),
            mo.hstack(
                [oversampling_control,
                 help_tip("Output bins per native sampling interval. Preset on load so a full sum fits the bin budget."),
                 fixed_step_control,
                 help_tip("Used only for the Fixed-Da grid.")],
                widths=[0.44, 0.06, 0.44, 0.06], gap=0.2, align="center",
            ),
            mo.hstack(
                [gap_factor_control,
                 help_tip("Break integration across gaps larger than this multiple of native spacing."),
                 max_bins_control,
                 help_tip("Ceiling on common-grid bins. Raise it for wide, high-resolution ranges; each bin costs ~24 bytes.")],
                widths=[0.44, 0.06, 0.44, 0.06], gap=0.2, align="center",
            ),
        ],
        gap=0.5,
    )

    _nowrap = {"white-space": "nowrap"}
    controls_card = card(
        mo.Html(
                "<div style='display:flex;align-items:center;gap:10px;margin:10px 0 2px;color:#64748b;font-size:0.72rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;white-space:nowrap;'>"
                "<span>Interaction</span>"
                "<span style='flex:1 1 auto;height:1px;background:#e2e8f0;'></span></div>"
            ),
        mo.hstack(
            [
                interaction_mode.style(_nowrap),
                spectrum_mode.style(_nowrap),
                ms_level_control.style(_nowrap),
            ],
            justify="start", wrap=True, gap=1.0, align="end",
        ),
        _settings,
        title="View & summation",
        subtitle="Click a TIC point for one scan, box-select a time window to sum, or sum all.",
    )
    # Kept above the plots so the primary action stays reachable; nowrap stops
    # the label folding inside the button.
    top_actions = mo.hstack(
        [sum_all_button.style(_nowrap)], justify="start", gap=0.5, align="center"
    )
    return (
        controls_card,
        fixed_step_control,
        gap_factor_control,
        grid_mode_control,
        interaction_mode,
        max_bins_control,
        ms_level_control,
        native_resolution,
        mz_max_control,
        mz_min_control,
        oversampling_control,
        profile_method_control,
        spectrum_mode,
        sum_all_button,
        top_actions,
    )


@app.cell
def _(
    COLOR_TIC,
    FONT_STACK,
    PLOT_TEMPLATE,
    get_source,
    go,
    interaction_mode,
    mo,
    np,
    reduce_xy_minmax,
):
    # --- Total ion chromatogram (blue) ---------------------------------------
    _source = get_source()
    tic_display_indices = np.array([], dtype=np.int64)
    tic_plot = None
    if _source is not None:
        _x, _y, tic_display_indices = reduce_xy_minmax(
            _source.times_min, _source.tic, max_points=20_000
        )
        _custom = np.column_stack(
            [tic_display_indices, _source.scan_numbers[tic_display_indices]]
        )
        _figure = go.Figure(
            go.Scattergl(
                x=_x, y=_y, mode="lines+markers",
                marker={"size": 4, "color": COLOR_TIC},
                line={"width": 1.2, "color": COLOR_TIC},
                customdata=_custom,
                hovertemplate=(
                    "Time: %{x:.6f} min<br>Total ion current: %{y:.6g}"
                    "<br>Scan: %{customdata[1]}<extra></extra>"
                ),
                name="TIC",
            )
        )
        _figure.update_layout(
            template=PLOT_TEMPLATE,
            title={"text": "Total ion chromatogram", "x": 0.0, "xanchor": "left", "y": 0.97},
            xaxis_title="Retention time (min)",
            yaxis_title="Total ion current",
            dragmode=interaction_mode.value,
            clickmode="event+select",
            hovermode="closest",
            height=280,
            margin={"l": 70, "r": 20, "t": 64, "b": 50},
            uirevision=f"{_source.name}-{interaction_mode.value}",
            font={"family": FONT_STACK},
            showlegend=True,
            legend={
                "orientation": "h", "x": 1, "xanchor": "right",
                "y": 1.02, "yanchor": "bottom",
                "bgcolor": "rgba(255,255,255,0)",
            },
        )
        tic_plot = mo.ui.plotly(
            _figure,
            config={
                "displaylogo": False, "scrollZoom": True,
                "modeBarButtonsToAdd": ["select2d"],
                "modeBarButtonsToRemove": ["lasso2d"],
                "responsive": True,
            },
        )
    return tic_display_indices, tic_plot


@app.cell
def _(np, set_view_request, tic_display_indices, tic_plot):
    # --- TIC selection -> view request ---------------------------------------
    if tic_plot is not None:
        _range = tic_plot.ranges.get("x")
        if _range and len(_range) == 2:
            set_view_request(
                {
                    "kind": "range",
                    "time_min": float(min(_range)),
                    "time_max": float(max(_range)),
                    "source": "tic_box_selection",
                }
            )
        elif tic_plot.indices:
            _display_index = int(tic_plot.indices[-1])
            if 0 <= _display_index < len(tic_display_indices):
                set_view_request(
                    {
                        "kind": "scan",
                        "index": int(tic_display_indices[_display_index]),
                        "source": "tic_point_selection",
                    }
                )
    return


@app.cell
def _(
    fixed_step_control,
    gap_factor_control,
    get_source,
    get_view_request,
    grid_mode_control,
    max_bins_control,
    mo,
    ms_level_control,
    mz_max_control,
    mz_min_control,
    np,
    oversampling_control,
    profile_method_control,
):
    # --- Resolve the requested spectrum (single scan / range / all) ----------
    _source = get_source()
    spectrum_mz = np.array([], dtype=float)
    spectrum_intensity = np.array([], dtype=float)
    spectrum_bin_edges = np.array([], dtype=float)
    spectrum_bin_area = np.array([], dtype=float)
    spectrum_error = None
    view_title = "No spectrum selected"
    view_caption = ""
    view_mode = "unknown"
    view_signal_label = "Intensity"

    if _source is not None:
        _request = get_view_request()
        _kind = _request.get("kind", "scan")
        try:
            if _kind == "scan":
                _index = int(np.clip(int(_request.get("index", 0)), 0, _source.n_scans - 1))
                _record = _source.read_spectrum(_index)
                spectrum_mz = _record.mz
                spectrum_intensity = _record.intensity
                view_mode = _record.mode
                view_title = f"Scan {_record.scan_number} — {_record.time_min:.4f} min"
                view_caption = (
                    f"MS{_record.ms_level}; {_record.mode}; {_record.mz.size:,} native points"
                )
            else:
                if _kind == "range":
                    _lo = float(_request.get("time_min", _source.time_min))
                    _hi = float(_request.get("time_max", _source.time_max))
                    _indices = _source.indices_in_time_range(_lo, _hi)
                    view_title = f"Summed — {_lo:.4f} to {_hi:.4f} min"
                else:
                    _indices = np.arange(_source.n_scans, dtype=np.int64)
                    view_title = "Summed — complete acquisition"
                if _indices.size == 0:
                    raise ValueError("No scans lie in the selected retention-time interval")
                _lv = int(ms_level_control.value)
                _level = None if _lv == 0 else _lv
                with mo.status.spinner(title="Streaming, rebinning, and summing spectra"):
                    _sum = _source.sum_spectra(
                        _indices,
                        mz_min=float(mz_min_control.value),
                        mz_max=float(mz_max_control.value),
                        ms_level=_level,
                        grid_mode=str(grid_mode_control.value),
                        oversampling=float(oversampling_control.value),
                        fixed_step=float(fixed_step_control.value),
                        profile_method=str(profile_method_control.value),
                        gap_factor=float(gap_factor_control.value),
                        max_grid_points=int(max_bins_control.value),
                    )
                spectrum_mz = _sum.mz
                spectrum_intensity = _sum.intensity
                spectrum_bin_edges = _sum.bin_edges
                spectrum_bin_area = _sum.bin_area
                view_mode = "summed"
                _level_text = "all MS levels" if _level is None else f"MS{_level}"
                _grid = "fixed-Da grid"
                if _sum.grid_mode == "auto_resolution":
                    _grid = (
                        f"constant-resolution grid; R≈{_sum.sampling_resolution:,.0f}; "
                        f"{_sum.oversampling:g}× oversampling"
                    )
                view_caption = (
                    f"{_sum.included_count:,} {_level_text} spectra of {_sum.requested_count:,} "
                    f"requested; {_grid}; {_sum.mz.size:,} bins"
                )
                if _sum.method == "area_corrected":
                    view_signal_label = "Summed intensity density"
        except Exception as exc:  # noqa: BLE001 - surface to the plot area
            spectrum_error = str(exc)
            # The bin-cap message tells you *that* the grid is too big but not
            # what to do about it. Bins scale linearly with oversampling, so
            # solve for the value that would fit and say so outright.
            if "bins" in spectrum_error and "grid" in spectrum_error:
                try:
                    _cap = int(max_bins_control.value)
                    _need = int(
                        "".join(ch for ch in spectrum_error.split("has")[1].split("bins")[0] if ch.isdigit())
                    )
                    _ov = float(oversampling_control.value)
                    # Bins scale linearly with oversampling. Take 2% headroom and
                    # floor to 1 dp, so the suggested value genuinely fits rather
                    # than landing exactly on the cap and rounding back over it.
                    _fits = int(_ov * _cap / _need * 0.98 * 10) / 10.0
                    _fits = max(1.0, _fits)
                    _hint = (
                        f" Oversampling {_fits:.1f} or lower would fit inside "
                        f"{_cap:,} bins at these m/z limits"
                        f" (currently {_ov:g})."
                        " Raise 'Max grid bins' instead if you need the finer grid"
                        " and have the memory."
                        " Integrated window intensities are area-conserving, so a"
                        " coarser grid barely changes them."
                    )
                    spectrum_error = spectrum_error + _hint
                except Exception:  # noqa: BLE001 - hint is best-effort only
                    pass
    return (
        spectrum_bin_area,
        spectrum_bin_edges,
        spectrum_error,
        spectrum_intensity,
        spectrum_mz,
        view_caption,
        view_mode,
        view_signal_label,
        view_title,
    )


@app.cell
def _(
    COLOR_MS,
    COLOR_WIN,
    FONT_STACK,
    PLOT_TEMPLATE,
    active_windows,
    get_mass_windows,
    go,
    mo,
    reduce_xy_minmax,
    spectrum_error,
    spectrum_intensity,
    spectrum_mode,
    spectrum_mz,
    view_mode,
    view_signal_label,
    view_title,
):
    # --- Mass spectrum (orange) with shaded extraction windows ---------------
    if spectrum_error:
        spectrum_plot = mo.callout(
            mo.md(f"**Could not build the spectrum**\n\n`{spectrum_error}`"), kind="danger"
        )
    elif spectrum_mz.size:
        _mz, _inten, _idx = reduce_xy_minmax(spectrum_mz, spectrum_intensity, max_points=30_000)
        _mode = "markers" if view_mode == "centroid" else "lines"
        _figure = go.Figure(
            go.Scattergl(
                x=_mz, y=_inten, mode=_mode,
                line={"width": 1.0, "color": COLOR_MS},
                marker={"size": 4, "color": COLOR_MS},
                hovertemplate="m/z: %{x:.6f}<br>Intensity: %{y:.6g}<extra></extra>",
                name="Mass spectrum",
            )
        )
        # Shade active windows (no in-plot text -> no title/label overlap).
        for _label, _low, _high in active_windows(get_mass_windows()):
            _figure.add_vrect(
                x0=_low, x1=_high, fillcolor=COLOR_WIN, opacity=0.12,
                line_width=0, layer="below",
            )
        _figure.update_layout(
            template=PLOT_TEMPLATE,
            title={"text": view_title, "x": 0.0, "xanchor": "left", "y": 0.97},
            xaxis_title="m/z",
            yaxis_title=view_signal_label,
            dragmode=spectrum_mode.value,
            clickmode="event+select",
            hovermode="closest",
            height=430,
            margin={"l": 70, "r": 20, "t": 64, "b": 55},
            font={"family": FONT_STACK},
            showlegend=True,
            legend={
                "orientation": "h", "x": 1, "xanchor": "right",
                "y": 1.02, "yanchor": "bottom",
                "bgcolor": "rgba(255,255,255,0)",
            },
        )
        spectrum_plot = mo.ui.plotly(
            _figure,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
                "modeBarButtonsToAdd": ["select2d"],
            },
        )
    else:
        spectrum_plot = mo.callout("Select a scan or time window on the TIC.", kind="info")
    return (spectrum_plot,)


@app.cell
def _(get_mass_windows, json, mo, pd, set_mass_windows, set_windows_touched):
    # --- Window editor + parameter save / load -------------------------------
    def _records(value):
        frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
        return frame.to_dict(orient="records")

    def _edited(value):
        set_windows_touched(True)
        set_mass_windows(_records(value))

    def _add(_value):
        set_windows_touched(True)

        def append(rows):
            rows = list(rows)
            if rows:
                try:
                    low = float(rows[-1]["m/z minimum"]) + 1.0
                    high = float(rows[-1]["m/z maximum"]) + 1.0
                except Exception:
                    low, high = 500.0, 501.0
            else:
                low, high = 500.0, 501.0
            rows.append(
                {
                    "Use": True,
                    "Label": f"Window {len(rows) + 1}",
                    "m/z minimum": round(low, 4),
                    "m/z maximum": round(high, 4),
                }
            )
            return rows

        set_mass_windows(append)

    def _remove(_value):
        set_windows_touched(True)
        set_mass_windows(lambda rows: list(rows[:-1]) if len(rows) > 1 else list(rows))

    def _load_params(files):
        if not files:
            return
        try:
            payload = json.loads(bytes(files[0].contents).decode("utf-8"))
            rows = payload.get("windows", payload) if isinstance(payload, dict) else payload
            cleaned = []
            for row in rows:
                cleaned.append(
                    {
                        "Use": bool(row.get("Use", True)),
                        "Label": str(row.get("Label", "Window")),
                        "m/z minimum": float(row["m/z minimum"]),
                        "m/z maximum": float(row["m/z maximum"]),
                    }
                )
            if cleaned:
                set_windows_touched(True)
                set_mass_windows(cleaned)
        except Exception:
            # Malformed parameter files are ignored rather than breaking the app.
            pass

    mass_window_editor = mo.ui.data_editor(
        pd.DataFrame(get_mass_windows()),
        label="",
        editable_columns="all",
        on_change=_edited,
    )
    add_window_button = mo.ui.button(label="Add window", kind="success", on_click=_add)
    remove_window_button = mo.ui.button(label="Remove last", on_click=_remove)
    save_windows_button = mo.download(
        data=lambda: json.dumps(
            {"windows": get_mass_windows()}, indent=2
        ).encode("utf-8"),
        filename="extraction_windows.json",
        mimetype="application/json",
        label="Save windows (.json)",
    )
    load_windows_widget = mo.ui.file(
        filetypes=[".json"], kind="button",
        label="Load windows", on_change=_load_params,
    )
    return (
        add_window_button,
        load_windows_widget,
        mass_window_editor,
        remove_window_button,
        save_windows_button,
    )


@app.cell
def _(
    active_windows,
    get_mass_windows,
    integrate_window,
    mo,
    np,
    spectrum_bin_area,
    spectrum_bin_edges,
    spectrum_intensity,
    spectrum_mz,
):
    # --- Per-window statistics for the current spectrum ----------------------
    _rows = []
    _has_area = (
        spectrum_bin_area.size > 0
        and spectrum_bin_edges.size == spectrum_bin_area.size + 1
    )
    if _has_area:
        _total = float(np.nansum(spectrum_bin_area))
    elif spectrum_mz.size > 1:
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        _total = float(_trapz(spectrum_intensity, spectrum_mz))
    else:
        _total = float(np.nansum(spectrum_intensity))

    for _label, _low, _high in active_windows(get_mass_windows()):
        if _has_area:
            _left = spectrum_bin_edges[:-1]
            _right = spectrum_bin_edges[1:]
            _width = _right - _left
            _overlap = np.maximum(
                0.0, np.minimum(_right, _high) - np.maximum(_left, _low)
            )
            _frac = np.divide(
                _overlap, _width, out=np.zeros_like(_overlap), where=_width > 0
            )
            _signal = float(np.nansum(spectrum_bin_area * _frac))
        else:
            _signal = integrate_window(spectrum_mz, spectrum_intensity, _low, _high)

        _mask = (spectrum_mz >= _low) & (spectrum_mz <= _high)
        if np.any(_mask):
            _x = spectrum_mz[_mask]
            _y = spectrum_intensity[_mask]
            _finite = np.isfinite(_y)
            if np.any(_finite):
                _fi = np.flatnonzero(_finite)
                _loc = int(_fi[int(np.argmax(_y[_finite]))])
                _bp_mz, _bp_int = float(_x[_loc]), float(_y[_loc])
            else:
                _bp_mz, _bp_int = np.nan, np.nan
        else:
            _bp_mz, _bp_int = np.nan, np.nan

        _pct = 100.0 * _signal / _total if _total > 0 else 0.0
        _rows.append(
            {
                "Window": _label,
                "m/z range": f"{_low:.4f}–{_high:.4f}",
                "Integrated signal": f"{_signal:.6g}",
                "% of total": f"{_pct:.4f}%",
                "Base-peak m/z": "—" if not np.isfinite(_bp_mz) else f"{_bp_mz:.6f}",
                "Base-peak intensity": "—" if not np.isfinite(_bp_int) else f"{_bp_int:.6g}",
            }
        )

    stats_table = mo.ui.table(
        _rows, pagination=False, selection=None, label="",
    )
    return (stats_table,)


@app.cell
def _(active_windows, get_mass_windows, mo):
    # --- Choose which m/z range of the displayed spectrum to export ----------
    _wins = active_windows(get_mass_windows())
    _options = {"Full displayed spectrum": None}
    for _lbl, _lo, _hi in _wins:
        _options[f"{_lbl}  ({_lo:.4f}–{_hi:.4f})"] = (_lo, _hi)
    export_window_select = mo.ui.dropdown(
        options=_options,
        value="Full displayed spectrum",
        label="Range",
    )
    return (export_window_select,)


@app.cell
def _(
    export_window_select,
    mo,
    np,
    pd,
    spectrum_intensity,
    spectrum_mz,
    view_signal_label,
):
    # --- Build the CSV for the current spectrum, clipped to the chosen range --
    def _spectrum_csv():
        mz = np.asarray(spectrum_mz, dtype=float)
        inten = np.asarray(spectrum_intensity, dtype=float)
        bounds = export_window_select.value
        if bounds is not None:
            low, high = bounds
            mask = (mz >= low) & (mz <= high)
            mz, inten = mz[mask], inten[mask]
        frame = pd.DataFrame({"mz": mz, view_signal_label: inten})
        return frame.to_csv(index=False).encode("utf-8")

    export_spectrum_button = mo.download(
        data=_spectrum_csv,
        filename="mass_spectrum.csv",
        mimetype="text/csv",
        label="Download CSV",
    )
    return (export_spectrum_button,)


@app.cell
def _(card, export_spectrum_button, export_window_select, mo, spectrum_mz):
    # --- Export card ----------------------------------------------------------
    if spectrum_mz.size:
        _body = mo.hstack(
            [export_window_select, export_spectrum_button],
            justify="start", wrap=True, gap=0.6, align="end",
        )
    else:
        _body = mo.md("Select a scan or time window to enable export.")
    export_card = card(
        _body,
        title="Export spectrum (CSV)",
        subtitle="Saves the displayed spectrum as <code>mz, intensity</code>, "
        "optionally clipped to one window.",
    )
    return (export_card,)


@app.cell
def _(mo, set_mass_windows, set_windows_touched, spectrum_plot):
    # --- Capture the current m/z selection straight into the window table -----
    # marimo's plotly integration reports *selection* ranges (box/lasso), not
    # zoom/relayout ranges, so this reads the box-selected span on the spectrum.
    def _selected_span():
        _ranges = getattr(spectrum_plot, "ranges", None) or {}
        _x = _ranges.get("x")
        if not _x or len(_x) < 2:
            return None
        try:
            low, high = sorted((float(_x[0]), float(_x[1])))
        except (TypeError, ValueError):
            return None
        if not (high > low):
            return None
        return low, high

    current_span = _selected_span()

    def _add_span(_value):
        span = _selected_span()
        if span is None:
            return
        low, high = span

        def append(rows):
            rows = list(rows)
            rows.append(
                {
                    "Use": True,
                    "Label": f"Window {len(rows) + 1}",
                    "m/z minimum": round(low, 4),
                    "m/z maximum": round(high, 4),
                }
            )
            return rows

        set_windows_touched(True)
        set_mass_windows(append)

    add_selection_button = mo.ui.button(
        label="Add current selection",
        kind="success",
        disabled=current_span is None,
        tooltip=(
            "Append the m/z span you box-selected on the spectrum "
            "as a new extraction window."
        ),
        on_click=_add_span,
    )
    if current_span is None:
        selection_hint = mo.Html(
            "<div class='card-sub'>Set <b>Spectrum drag</b> to "
            "<i>Select an m/z window</i>, then drag across the spectrum to "
            "capture a span.</div>"
        )
    else:
        selection_hint = mo.Html(
            f"<div class='card-sub'>Selected span: "
            f"<b>{current_span[0]:.4f} – {current_span[1]:.4f}</b></div>"
        )
    return add_selection_button, selection_hint


@app.cell
def _(
    add_selection_button,
    add_window_button,
    card,
    load_windows_widget,
    mass_window_editor,
    mo,
    remove_window_button,
    save_windows_button,
    selection_hint,
    stats_table,
):
    windows_card = card(
        mo.hstack(
            [add_window_button, add_selection_button, remove_window_button],
            justify="start", gap=0.5,
        ),
        selection_hint,
        mass_window_editor.style(
            {
                "max-height": "230px",
                "overflow": "auto",
                "width": "100%",
                "min-width": "0",
                "max-width": "100%",
            }
        ),
        mo.hstack([save_windows_button, load_windows_widget], justify="start", gap=0.5),
        title="Extraction windows",
        subtitle="Kept when you load another file. Save/load a set as JSON for a session.",
    )
    stats_card = card(
        stats_table.style(
            {
                "max-height": "300px",
                "overflow": "auto",
                "width": "100%",
                "min-width": "0",
                "max-width": "100%",
            }
        ),
        title="Window statistics — current spectrum",
        subtitle="Area-corrected sums use exact bin overlap at window edges.",
    )
    return stats_card, windows_card


@app.cell
def _(
    fit_oversampling,
    fixed_step_control,
    get_source,
    grid_mode_control,
    max_bins_control,
    mo,
    mz_max_control,
    mz_min_control,
    native_resolution,
    oversampling_control,
    predict_bins,
):
    # --- Live grid budget -----------------------------------------------------
    # Predicts the bin count the reader would build from the current settings, so
    # an over-budget "Sum all" is visible before it is pressed rather than after
    # it fails. Same formula the reader uses, so the two cannot disagree.
    if get_source() is None:
        grid_estimate = mo.md("")
    else:
        _lo = float(mz_min_control.value)
        _hi = float(mz_max_control.value)
        _cap = int(max_bins_control.value)
        _ov = float(oversampling_control.value)
        _bins = predict_bins(
            str(grid_mode_control.value), _lo, _hi, native_resolution,
            _ov, float(fixed_step_control.value),
        )
        _res = (
            f"native sampling R &asymp; {native_resolution:,.0f}"
            if native_resolution
            else "native sampling resolution could not be estimated"
        )
        if _bins is None:
            grid_estimate = mo.callout(mo.md(f"Grid size cannot be predicted ({_res})."), kind="neutral")
        elif _bins <= _cap:
            grid_estimate = mo.callout(
                mo.md(
                    f"**Grid budget OK.** A full sum over {_lo:,.0f}\u2013{_hi:,.0f} "
                    f"would build **{_bins:,} bins** (~{_bins * 24 / 1024**2:,.0f} MB), "
                    f"within the {_cap:,} ceiling. <span class='card-sub'>{_res}.</span>"
                ),
                kind="success",
            )
        else:
            _needed = int(_bins * 1.02)
            if str(grid_mode_control.value) == "auto_resolution":
                _fits = fit_oversampling(_lo, _hi, native_resolution, _cap, preferred=_ov)
                _at_fits = predict_bins(
                    "auto_resolution", _lo, _hi, native_resolution, _fits, 1.0
                )
                if _at_fits is not None and _at_fits <= _cap:
                    _advice = (
                        f"Drop **Oversampling** to **{_fits:g}** (currently {_ov:g}), "
                        f"or raise **Max grid bins** to about **{_needed:,}**"
                    )
                else:
                    # Oversampling floors at 1, so it cannot always rescue the grid.
                    _floor = predict_bins(
                        "auto_resolution", _lo, _hi, native_resolution, 1.0, 1.0
                    )
                    _advice = (
                        "Oversampling cannot fix this on its own — even at **1.0** the grid "
                        f"needs **{_floor:,} bins**. Raise **Max grid bins** to at least "
                        f"**{int(_floor * 1.02):,}**, or narrow the m/z limits"
                    )
            else:
                _advice = f"Widen the fixed step, or raise **Max grid bins** to about **{_needed:,}**"
            grid_estimate = mo.callout(
                mo.md(
                    f"**A full sum would fail.** It needs **{_bins:,} bins**, over the "
                    f"{_cap:,} ceiling. {_advice}. Integrated window intensities are "
                    "area-conserving, so a coarser grid barely changes them. "
                    f"<span class='card-sub'>{_res}.</span>"
                ),
                kind="warn",
            )
    return (grid_estimate,)


@app.cell
def _(
    controls_card,
    export_card,
    get_source,
    grid_estimate,
    mo,
    spectrum_plot,
    stats_card,
    tic_plot,
    top_actions,
    view_caption,
    windows_card,
):
    # --- Explore tab layout ---------------------------------------------------
    # marimo's hstack `widths` only sets flex-grow; the wide data editor keeps
    # the right column at its min-content size and it bleeds past the container.
    # Giving each column a definite width makes the overflow scroll internally.
    _col = {
        "min-width": "0",
        "overflow-x": "auto",
        "box-sizing": "border-box",
    }
    if get_source() is None:
        explore_panel = mo.callout(
            "Load an mzML file in the **Load** tab to begin.", kind="info"
        )
    else:
        _left = mo.vstack(
            [
                top_actions,
                tic_plot,
                spectrum_plot,
                mo.Html(f"<div class='plot-note'>{view_caption}</div>") if view_caption else mo.md(""),
                controls_card,
                grid_estimate,
            ],
            gap=0.6,
        ).style({**_col, "width": "61%"})
        _right = mo.vstack(
            [windows_card, stats_card, export_card], gap=0.7
        ).style({**_col, "width": "37%"})
        explore_panel = mo.hstack(
            [_left, _right], justify="start", gap=0.8, align="start"
        )
    return (explore_panel,)


@app.cell
def _(
    active_windows,
    get_mass_windows,
    load_uploaded_source,
    mo,
    pd,
    set_batch_result,
    stream_integrate,
):
    # --- Batch extraction: many files x current windows ----------------------
    batch_files_widget = mo.ui.file(
        filetypes=[".mzML", ".mzml"], multiple=True, kind="area",
        label="Add one or more .mzML files",
    )
    batch_level_control = mo.ui.dropdown(
        options={"MS1": 1, "MS2": 2, "MS3": 3, "All levels": 0},
        value="MS1", label="Level summed per file",
    )

    def _run_batch(_value):
        files = batch_files_widget.value
        windows = active_windows(get_mass_windows())
        if not files or not windows:
            set_batch_result(
                {"error": "Add at least one file and define at least one active window."}
            )
            return
        level = int(batch_level_control.value)
        records = []
        errors = []
        for upload in files:
            try:
                source = load_uploaded_source(upload)
                import numpy as _np

                indices = _np.arange(source.n_scans, dtype=_np.int64)
                totals, grand = stream_integrate(source, indices, windows, level)
                row = {"File": source.name, "Spectra": source.n_scans}
                for label, _lo, _hi in windows:
                    row[label] = totals[label]
                for label, _lo, _hi in windows:
                    row[f"{label} (% total)"] = (
                        100.0 * totals[label] / grand if grand > 0 else 0.0
                    )
                records.append(row)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{getattr(upload, 'name', 'file')}: {exc}")
        set_batch_result(
            {
                "frame": pd.DataFrame(records) if records else None,
                "errors": errors,
                "windows": [f"{lb} [{lo:.4f}–{hi:.4f}]" for lb, lo, hi in windows],
            }
        )

    batch_run_button = mo.ui.button(
        label="Run extraction",
        kind="success",
        tooltip="Stream every file once and integrate each active window.",
        on_click=_run_batch,
    )
    return batch_files_widget, batch_level_control, batch_run_button


@app.cell
def _(card, get_batch_result, mo, pd):
    # --- Batch results + CSV download ----------------------------------------
    _result = get_batch_result()
    if _result is None:
        batch_results_card = card(
            mo.md("Add files and press **Run extraction** to build the table."),
            title="Results",
        )
    elif _result.get("error"):
        batch_results_card = card(
            mo.callout(_result["error"], kind="warn"), title="Results"
        )
    else:
        _frame = _result.get("frame")
        _errors = _result.get("errors") or []
        _parts = []
        if _frame is not None and not _frame.empty:
            _parts.append(
                mo.ui.table(_frame, pagination=False, selection=None, label="").style(
                    {"max-height": "360px", "overflow": "auto"}
                )
            )
            _parts.append(
                mo.download(
                    data=_frame.to_csv(index=False).encode("utf-8"),
                    filename="window_intensities.csv",
                    mimetype="text/csv",
                    label="Download CSV",
                )
            )
        else:
            _parts.append(mo.md("No files produced results."))
        if _errors:
            _parts.append(
                mo.callout(mo.md("**Skipped:**  \n" + "  \n".join(_errors)), kind="warn")
            )
        batch_results_card = card(*_parts, title="Results")
    return (batch_results_card,)


@app.cell
def _(
    active_windows,
    batch_files_widget,
    batch_level_control,
    batch_results_card,
    batch_run_button,
    card,
    get_mass_windows,
    mo,
):
    # --- Batch tab layout -----------------------------------------------------
    _windows = active_windows(get_mass_windows())
    _summary = (
        ", ".join(f"{lb} [{lo:.2f}–{hi:.2f}]" for lb, lo, hi in _windows)
        if _windows
        else "No active windows yet — define them in the Explore & windows tab."
    )
    batch_panel = mo.vstack(
        [
            card(
                batch_files_widget,
                mo.hstack([batch_level_control, batch_run_button], justify="start",
                          gap=0.6, align="end"),
                title="Batch extraction",
                subtitle="Applies the current extraction windows to every uploaded file.",
            ),
            card(
                mo.Html(f"<div class='card-sub'>Active windows: {_summary}</div>"),
                title="Windows to extract",
            ),
            batch_results_card,
        ],
        gap=0.8,
    )
    return (batch_panel,)


@app.cell
def _(
    app_css,
    batch_panel,
    explore_panel,
    get_source,
    load_panel,
    mo,
    title_banner,
):
    _default = "Explore & windows" if get_source() is not None else "Load"
    _tabs = mo.ui.tabs(
        {
            "Load": load_panel,
            "Explore & windows": explore_panel,
            "Batch extract": batch_panel,
        },
        value=_default,
    )
    mo.vstack(
        [
            app_css,
            title_banner("Peakachu", "mzML window intensity viewer"),
            _tabs,
        ],
        gap=0.4,
    )
    return


if __name__ == "__main__":
    app.run()
