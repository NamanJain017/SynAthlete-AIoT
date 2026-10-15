"""Tests for the SoccerMon/PMData readers and the 21-day episode rule (piece 2, D-018)."""

import pandas as pd

from src.data import daily, readers


def _days(n: int = 120) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def _day(i: int) -> pd.Timestamp:
    return pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)


def test_reports_within_21_days_are_one_episode_and_22_days_start_a_new_one():
    labels = daily.label_days(_days(), {_day(30), _day(51), _day(73)})
    starts = labels.index[labels["episode_start"]].tolist()
    assert starts == [_day(30), _day(73)]  # 30 -> 51 is 21 days (same), 51 -> 73 is 22 (new)
    assert labels["pain_report"].sum() == 3


def test_first_report_close_to_the_data_start_is_not_a_new_episode():
    labels = daily.label_days(_days(), {_day(10), _day(60)})
    assert labels.index[labels["episode_start"]].tolist() == [_day(60)]


def test_healthy_days_are_far_from_reports_and_from_the_data_edges():
    labels = daily.label_days(_days(120), {_day(60)})
    healthy = labels["healthy_event"]
    assert not healthy[_day(39):_day(81)].any()  # within 21 days of the report
    assert not healthy[:_day(21)].any() and not healthy[_day(98):].any()  # too close to the edges
    assert healthy[_day(22):_day(38)].all()


def test_pmdata_injury_reader_ignores_empty_check_ins(tmp_path):
    path = tmp_path / "injury.csv"
    path.write_text('effective_time_frame,injuries\n'
                    '2019-11-07T06:39:48.428Z,{}\n'
                    '2020-01-07T23:30:00.000Z,"{\'right_hand\': \'minor\'}"\n', encoding="utf-8")
    df = readers.read_pmdata_injuries(path)
    assert df["has_injury"].tolist() == [False, True]
    assert df["date"].tolist() == [pd.Timestamp("2019-11-07"), pd.Timestamp("2020-01-07")]


def test_soccermon_wide_reader_handles_any_first_header_and_day_first_dates(tmp_path):
    path = tmp_path / "fatigue.csv"
    path.write_text("Fatigue Data,TeamA-1,TeamB-2\n02.01.2020,3,\n13.01.2020,,4\n",
                    encoding="utf-8")
    df = readers.read_soccermon_wide(path)
    assert df.index.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-13")]
    assert df.loc[pd.Timestamp("2020-01-13"), "TeamB-2"] == 4


def test_fitbit_reader_handles_plain_and_nested_values(tmp_path):
    nested = tmp_path / "rhr.json"
    nested.write_text(
        '[{"dateTime": "2019-11-01 00:00:00", "value": {"value": 53.7, "error": 6.8}}]',
        encoding="utf-8",
    )
    plain = tmp_path / "steps.json"
    plain.write_text('[{"dateTime": "2019-11-01 00:01:00", "value": "12"}]', encoding="utf-8")
    assert readers.read_fitbit_json(nested)["value"].tolist() == [53.7]
    assert readers.read_fitbit_json(plain)["value"].tolist() == [12]
