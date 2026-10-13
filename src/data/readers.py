"""Turn one raw dataset file into Python objects.

No cleaning, resampling or windowing here: the audit (piece 1) and the loaders (piece 2) both
build on these readers, so each raw format is parsed in exactly one place.
"""

from __future__ import annotations

import pickle
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

# ---------------------------------------------------------------- runner logs (Lövdal / Kaggle)

RUNNER_KINDS = ("day", "week")
RUNNER_ID_COLS = ("Athlete ID", "injury", "Date")
RUNNER_SUFFIXES = (".csv", ".tab", ".tsv")  # DataverseNL may re-save CSVs as tab-separated .tab


def find_runner_files(folder: Path) -> dict[str, Path]:
    """Locate the day- and week-approach files by name, wherever they sit in the folder."""
    found: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in RUNNER_SUFFIXES:
            continue
        for kind in RUNNER_KINDS:
            if f"{kind}_approach" in path.name.lower() and kind not in found:
                found[kind] = path
    return found


def read_runner_csv(path: Path) -> pd.DataFrame:
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    return pd.read_csv(path, sep=sep)


# ---------------------------------------------------------------- PAMAP2

# Column layout from PAMAP2_Dataset/readme.pdf, section II.1 (54 columns per row).
PAMAP2_IMU_FIELDS = (
    "temp",
    "acc16_x", "acc16_y", "acc16_z",
    "acc6_x", "acc6_y", "acc6_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "orient_1", "orient_2", "orient_3", "orient_4",
)
PAMAP2_IMU_POSITIONS = ("hand", "chest", "ankle")
PAMAP2_COLUMNS = ["timestamp", "activity_id", "heart_rate"] + [
    f"{pos}_{field}" for pos in PAMAP2_IMU_POSITIONS for field in PAMAP2_IMU_FIELDS
]
# Activity IDs from readme.pdf, section II.2. 0 = transient, to be discarded.
PAMAP2_ACTIVITIES = {
    0: "transient", 1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "Nordic walking", 9: "watching TV", 10: "computer work", 11: "car driving",
    12: "ascending stairs", 13: "descending stairs", 16: "vacuum cleaning", 17: "ironing",
    18: "folding laundry", 19: "house cleaning", 20: "playing soccer", 24: "rope jumping",
}


def read_pamap2_dat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", header=None)
    if df.shape[1] != len(PAMAP2_COLUMNS):
        raise ValueError(f"{path.name}: expected {len(PAMAP2_COLUMNS)} columns, got {df.shape[1]}")
    df.columns = PAMAP2_COLUMNS
    return df


# ---------------------------------------------------------------- WESAD

# Documented in WESAD/wesad_readme.pdf, sections II.1-II.3.
WESAD_CHEST_HZ = 700
WESAD_WRIST_HZ = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
WESAD_LABELS = {
    0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation",
    5: "ignore", 6: "ignore", 7: "ignore",
}


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickling can run arbitrary code, so only the numpy pieces that build arrays are allowed."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refused {module}.{name}: only numpy arrays are allowed")
        # numpy 2 renamed numpy.core to numpy._core; WESAD pickles use the old name.
        return super().find_class(module.replace("numpy.core.", "numpy._core."), name)


def read_wesad_pickle(path: Path) -> dict:
    """Keys: 'subject', 'signal' {'chest': {...}, 'wrist': {...}}, 'label' (700 Hz)."""
    with open(path, "rb") as f:
        # The pickles were written by Python 2; latin1 keeps numpy byte buffers intact.
        return NumpyOnlyUnpickler(f, encoding="latin1").load()


def read_wesad_readme(path: Path) -> dict[str, str]:
    """Personal information block of SX_readme.txt, e.g. {'Age': '27', 'Gender': 'male'}."""
    keys = ("Age", "Height (cm)", "Weight (kg)", "Gender", "Dominant hand")
    info: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() in keys:
            info[key.strip()] = value.strip()
    return info


# ---------------------------------------------------------------- Sport DB 2.0

# Dem.txt header spellings differ between recordings; map them onto one name.
SPORTDB_DEM_ALIASES = {
    "Smoking": "Smoker",
    "Pre-workourt": "Diatery_supplement_pre_workout",
    "Post-workourt": "Diatery_supplement_post_workout",
}
_TRNOTE_PHASE = re.compile(
    r"^(?P<name>.*?)\s*:\s*from\s+(?P<start>\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"\s+to\s+(?P<end>\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)"
)


def hms_to_seconds(text: str) -> float:
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_trnote(text: str) -> list[tuple[str, float, float]]:
    """Timed phases ('Resting : from 00:00:00 to 00:18:30'); other note styles are skipped."""
    phases = []
    for line in text.splitlines():
        m = _TRNOTE_PHASE.match(line.strip())
        if m:
            phases.append(
                (m["name"].strip(), hms_to_seconds(m["start"]), hms_to_seconds(m["end"]))
            )
    return phases


def _dem_value(raw: str) -> float | str | None:
    if raw in ("", "NA"):
        return None
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_sportdb_dem(text: str) -> dict[str, float | str | None]:
    """Two-line Dem.txt (header + values). Values are tab-separated, sometimes mixed with spaces."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    header = [SPORTDB_DEM_ALIASES.get(h.strip(), h.strip()) for h in lines[0].split("\t")]
    values = lines[1].split("\t")
    if len(values) != len(header):
        values = lines[1].split()
    if len(values) != len(header):
        return {"_unparsed": lines[1]}
    return {h: _dem_value(v.strip()) for h, v in zip(header, values, strict=True)}


def _flatten_mat_struct(obj, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in obj._fieldnames:
        value = getattr(obj, name)
        if hasattr(value, "_fieldnames"):
            out.update(_flatten_mat_struct(value, f"{prefix}{name}."))
        else:
            out[f"{prefix}{name}"] = np.asarray(value)
    return out


def read_sportdb_record(crd_dir: Path) -> dict:
    """One recording folder (.../CRDn): signals from Data.mat, demographics, timed phases."""
    data = loadmat(crd_dir / "Data.mat", squeeze_me=True, struct_as_record=False)["Data"]
    dem_path, note_path = crd_dir / "Dem.txt", crd_dir / "TrNote.txt"
    return {
        "signals": _flatten_mat_struct(data),
        "dem": parse_sportdb_dem(dem_path.read_text(encoding="utf-8", errors="replace"))
        if dem_path.exists()
        else {},
        "phases": parse_trnote(note_path.read_text(encoding="utf-8", errors="replace"))
        if note_path.exists()
        else [],
    }


# ---------------------------------------------------------------- Kaggle tabular files


def read_sirp_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def xlsx_metadata(path: Path) -> dict[str, str]:
    """Creating application and timestamps from the .xlsx docProps (no personal names kept)."""
    meta: dict[str, str] = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if "docProps/core.xml" in names:
            core = z.read("docProps/core.xml").decode("utf-8", errors="replace")
            for tag in ("created", "modified"):
                m = re.search(rf"<dcterms:{tag}[^>]*>([^<]+)<", core)
                if m:
                    meta[tag] = m.group(1)
        if "docProps/app.xml" in names:
            app = z.read("docProps/app.xml").decode("utf-8", errors="replace")
            m = re.search(r"<Application>([^<]+)<", app)
            if m:
                meta["application"] = m.group(1)
    return meta


def read_ziya_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Record_ID": str})
