"""Indexed, on-demand mzML access for the marimo spectrum viewer.

The source file remains on disk.  Only the compact spectrum index, TIC arrays,
one decoded spectrum, and the current summed accumulator need to be resident in
memory.  The implementation supports uncompressed ``.mzML`` files, including
``indexedmzML`` files with random-access offsets.  Non-indexed mzML is accepted
locally by building spectrum offsets with a memory-mapped one-time scan.

Thanks Michael Marty and those supporting UniDec
Also leveraged lessons from pyteomics. 

"""

from __future__ import annotations

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
        with self.path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                position = 0
                while True:
                    start = mm.find(b"<spectrum ", position)
                    if start < 0:
                        break
                    end = mm.find(b"</spectrum>", start)
                    if end < 0:
                        raise MzMLIndexedReaderError("Unterminated <spectrum> element")
                    end += len(b"</spectrum>")
                    opening_end = mm.find(b">", start, min(end, start + 8192))
                    opening = bytes(mm[start : opening_end + 1]) if opening_end >= 0 else b""
                    identifier = _attrs(opening).get("id", f"spectrum={len(starts) + 1}")
                    starts.append(start)
                    ends.append(end)
                    identifiers.append(identifier)
                    position = end

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
