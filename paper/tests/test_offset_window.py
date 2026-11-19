"""offset_window: the rule that derives the window around chase offsets, and its reader."""

from __future__ import annotations

import json

import pytest

import manuscript_stats as ms
import offset_window as ow
import paper_config

PRIMARY = "primary_run"
REGION = [-5.299999999999649, 10.600000000000577]  # the primary run's region on 2026-11-19


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    monkeypatch.delenv(ow.OVERRIDE_ENV, raising=False)


def write_record(path, namespace, values, detections=PRIMARY):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"namespace": namespace, "script": "x.py", "detections": detections,
                                "commit": "c0ffee", "code_modified": False, "values": values}))


def write_event_rates(out, region):
    write_record(out / "stats" / "event_rates.json", "event_rates",
                 {"loser": {"chase_offset": {"peak": {"time_s": -0.4, "region_s": region}}}})


@pytest.fixture
def primary(tmp_path, monkeypatch):
    """The output directory of a primary run with event rates and its window."""
    monkeypatch.setattr(paper_config, "PRIMARY", PRIMARY)
    out = tmp_path / "output" / PRIMARY
    write_event_rates(out, REGION)
    write_record(out / "stats" / "offset_window.json", "offset_window", ow.window_stats(REGION))
    return out


# ------------------------------------------------------------------ the rule


@pytest.mark.parametrize(("region", "window"), [
    (REGION, (-5, 11)),                              # the primary run, 2026-11-19
    ([-3.0999999999996, 13.200000000000614], (-3, 13)),  # the earlier curve: the old window
    ([-0.6, 0.6], (-1, 1)),
    ([-5.4, 10.4], (-5, 10)),                        # nearest, not outward
    ([-5.6, 10.6], (-6, 11)),
    ([-5.5, 10.5], (-6, 11)),                        # half-way: outwards, widening the window
    ([-5.499999999999645, 10.500000000000576], (-6, 11)),  # half-way up to the float noise of the grid
    ([-5.4999, 10.5001], (-5, 11)),                  # not half-way
])
def test_derive_rounds_to_whole_seconds(region, window):
    assert ow.derive_window(region) == window


def test_derive_keeps_the_sign():
    left, right = ow.derive_window([-5.3, 10.6])
    assert (left, right) == (-5, 11) and isinstance(left, int) and isinstance(right, int)
    assert ow.derive_window((-12.7, 0.9)) == (-13, 1)


@pytest.mark.parametrize("region", [
    [0.6, 10.6],    # starts after the offset
    [-10.6, -0.6],  # ends before it
    [-0.4, 10.6],   # rounds to a window that starts at the offset
    [-5.3, 0.4],    # ... or ends there
])
def test_derive_refuses_a_window_without_the_offset(region):
    with pytest.raises(ValueError, match="does not contain the chase offset"):
        ow.derive_window(region)


@pytest.mark.parametrize("region", [
    None, -5.3, [], [-5.3], [-5.3, 10.6, 12.0], "[-5.3, 10.6]", ["-5.3", 10.6],
    [float("nan"), 10.6], [-5.3, float("inf")], [True, 10.6], {"start": -5.3, "end": 10.6},
])
def test_derive_refuses_a_malformed_region(region):
    with pytest.raises(ValueError, match="is not \\[start, end\\]"):
        ow.derive_window(region)


def test_derive_refuses_a_reversed_region():
    with pytest.raises(ValueError, match="does not end after it starts"):
        ow.derive_window([10.6, -5.3])


def test_window_stats_records_window_region_and_rule():
    assert ow.window_stats(REGION) == {"window_s": [-5, 11], "width_s": 16, "region_s": REGION,
                                       "rule": {"null_band": 0.95, "rounding_s": 1}}


# ------------------------------------------------------------------ the region


def test_region_is_read_from_event_rates(tmp_path):
    write_event_rates(tmp_path, REGION)
    assert ow.region_from_event_rates(tmp_path / "stats" / "event_rates.json") == REGION


def test_region_fails_on_missing_event_rates(tmp_path):
    with pytest.raises(FileNotFoundError, match="event_rates.json is missing"):
        ow.region_from_event_rates(tmp_path / "stats" / "event_rates.json")


def test_region_fails_without_a_loser_peak(tmp_path):
    path = tmp_path / "stats" / "event_rates.json"
    write_record(path, "event_rates", {"loser": {"chase_offset": {"before": {}}}})
    with pytest.raises(ValueError, match="no event_rates.loser.chase_offset.peak.region_s"):
        ow.region_from_event_rates(path)


# ------------------------------------------------------------------ the reader


def test_reader_returns_the_primary_window(primary):
    assert ow.offset_window(primary) == (-5.0, 11.0)


def test_reader_fails_without_the_primary_window(primary):
    (primary / "stats" / "offset_window.json").unlink()
    with pytest.raises(FileNotFoundError, match=f"offset_window.json is missing.*{PRIMARY}"):
        ow.offset_window(primary)


def test_reader_fails_when_the_event_rates_changed(primary):
    write_event_rates(primary, [-5.3, 12.6])  # rates rerun, window step not
    with pytest.raises(ValueError, match="no longer matches the primary's event rates"):
        ow.offset_window(primary)


def test_reader_fails_on_a_hand_edited_window(primary):
    stats = ow.window_stats(REGION)
    stats["window_s"] = [-3, 13]
    write_record(primary / "stats" / "offset_window.json", "offset_window", stats)
    with pytest.raises(ValueError, match="no longer matches"):
        ow.offset_window(primary)


def test_reader_fails_on_another_runs_window(primary):
    write_record(primary / "stats" / "offset_window.json", "offset_window",
                 ow.window_stats(REGION), detections="rtdetr_gapfill")
    with pytest.raises(ValueError, match="is not the window of the primary run"):
        ow.offset_window(primary)


def test_override_replaces_the_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ow.OVERRIDE_ENV, "-3,13")
    assert ow.offset_window(tmp_path / "nothing") == (-3.0, 13.0)
    assert "testing only" in capsys.readouterr().err


@pytest.mark.parametrize("text", ["13", "-3,13,1", "a,b", "3,13", "-3,-1", ""])
def test_override_must_be_a_window_around_the_offset(tmp_path, monkeypatch, text):
    monkeypatch.setenv(ow.OVERRIDE_ENV, text)
    with pytest.raises(ValueError):
        ow.offset_window(tmp_path)


def test_override_is_a_testing_override_of_the_stats():
    assert ow.OVERRIDE_ENV in ms.TESTING_OVERRIDES


# ------------------------------------------------------------------ the step


@pytest.fixture
def step(primary, monkeypatch):
    """Run offset_window.py's main for a run: step(detections) -> its stats file."""
    monkeypatch.setattr(ms, "_git", lambda *args: "" if args[0] == "status" else "c0ffee")
    monkeypatch.setattr(paper_config, "PRIMARY_OUT", primary)

    def run(detections, out):
        monkeypatch.setattr(paper_config, "DETECTIONS", detections)
        monkeypatch.setattr(paper_config, "OUT", out)
        monkeypatch.setattr(ow.sys, "argv", ["offset_window.py"])
        ow.main()
        return json.loads((out / "stats" / "offset_window.json").read_text())
    return run


def test_step_derives_the_window_in_the_primary_run(step, tmp_path):
    out = tmp_path / "scratch" / PRIMARY
    write_event_rates(out, [-3.1, 13.2])
    record = step(PRIMARY, out)
    assert record["values"]["window_s"] == [-3, 13]  # from this run's event rates
    assert record["script"] == "offset_window.py" and "testing_overrides" not in record


def test_step_copies_the_primary_window_in_a_sensitivity_run(step, tmp_path):
    out = tmp_path / "output" / "sensitivity"
    write_event_rates(out, [-8.0, 13.3])  # its own curve is not used
    record = step("sensitivity", out)
    assert record["values"] == ow.window_stats(REGION)
    assert record["detections"] == "sensitivity"


def test_step_fails_in_a_sensitivity_run_before_the_primary(step, primary, tmp_path):
    (primary / "stats" / "offset_window.json").unlink()
    with pytest.raises(FileNotFoundError, match="offset_window.json is missing"):
        step("sensitivity", tmp_path / "output" / "sensitivity")


def test_step_refuses_the_override(step, tmp_path, monkeypatch):
    monkeypatch.setenv(ow.OVERRIDE_ENV, "-3,13")
    with pytest.raises(SystemExit, match="takes no override"):
        step(PRIMARY, tmp_path / "output" / PRIMARY)
