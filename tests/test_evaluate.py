import pytest

from nexus.evaluate import evaluate, load_truth_csv, match_events, save_truth_csv


def test_perfect_match():
    truth = [(1.0, "IN"), (5.0, "OUT")]
    r = evaluate([(1.2, "IN"), (4.8, "OUT")], truth, tolerance_s=1.0)
    assert r.overall.true_positives == 2 and r.f1 == 1.0
    assert r.directions["IN"].count_error_pct == 0.0


def test_direction_must_match_and_each_event_matches_once():
    assert match_events([(1.0, "OUT")], [(1.0, "IN")], 1.0) == (0, 1, 1)
    assert match_events([(1.0, "IN")], [(1.0, "IN"), (1.2, "IN")], 1.0) == (1, 0, 1)


def test_tolerance_window():
    assert match_events([(3.0, "IN")], [(1.0, "IN")], 1.5) == (0, 1, 1)
    assert match_events([(2.4, "IN")], [(1.0, "IN")], 1.5) == (1, 0, 0)


def test_overcount_and_undercount_metrics():
    truth = [(t, "IN") for t in (1, 5, 9, 13)]
    pred = [(1, "IN"), (1.3, "IN"), (5, "IN"), (30, "IN"), (31, "IN")]
    r = evaluate(pred, truth, 1.0)
    s = r.directions["IN"]
    assert (s.true_positives, s.false_positives, s.false_negatives) == (2, 3, 2)
    assert s.precision == pytest.approx(0.4) and s.recall == pytest.approx(0.5)
    assert s.count_error_pct == pytest.approx(25.0)
    assert r.directions["OUT"].count_error_pct is None           # no OUT truth -> undefined


def test_empty_cases():
    r = evaluate([], [], 1.0)
    assert r.f1 is None and r.overall.precision is None
    r = evaluate([(1, "IN")], [], 1.0)
    assert r.overall.false_positives == 1 and r.overall.recall is None
    assert "TOTAL" in r.format_text()


def test_truth_csv_roundtrip_and_errors(tmp_path):
    p = tmp_path / "t.csv"
    save_truth_csv([(5.0, "OUT"), (1.25, "IN")], p)
    assert load_truth_csv(p) == [(1.25, "IN"), (5.0, "OUT")]
    p.write_text("time_s,direction\n1.0,SIDEWAYS\n")
    with pytest.raises(ValueError, match="row 2"):
        load_truth_csv(p)
    p.write_text("# comment\n2.0,in\n")                           # no header, lowercase ok
    assert load_truth_csv(p) == [(2.0, "IN")]
