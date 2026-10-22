"""Reference datasets (piece 2): Sport DB 2.0 subject grouping checks."""

from src.data import reference


def test_identical_demographics_files_are_grouped_by_subject_folder(tmp_path):
    for rec, dem in [("CYC/S1/CRD1", "same"), ("CYC/S2/CRD1", "same"), ("CYC/S2/CRD2", "same"),
                     ("RUN/S1/CRD1", "other"), ("SKA/CRD1", "same")]:
        (tmp_path / rec).mkdir(parents=True)
        (tmp_path / rec / "Data.mat").write_bytes(b"")
        (tmp_path / rec / "Dem.txt").write_text(dem, encoding="utf-8")
    mats = sorted(tmp_path.rglob("Data.mat"))
    assert reference.identical_dem_groups(mats, tmp_path) == [
        ["CYC/S1", "CYC/S2", "SKA/<no subject folder>"]]
