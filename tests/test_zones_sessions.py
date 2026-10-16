import pytest

from nexus.config import ConfigError, config_from_dict, default_config, sample_zones
from nexus.sessions import SessionTracker
from nexus.zones import ZoneMap, point_in_polygon

SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_in_polygon_inside_outside_edge_and_concave():
    assert point_in_polygon(5, 5, SQ) and not point_in_polygon(11, 5, SQ)
    assert point_in_polygon(10, 5, SQ)                                  # edge counts as inside
    ell = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]         # L shape
    assert point_in_polygon(2, 8, ell) and not point_in_polygon(8, 8, ell)


def test_zone_map_lookup_centroid_and_masks():
    zm = ZoneMap(sample_zones())
    assert zm.zone_at(100, 100) == "Display A" and zm.zone_at(250, 250) == "Display C"
    assert zm.zone_at(250, 450) == "Entrance" and zm.zone_at(5, 5) is None
    cx, cy = zm.centroid("Display A")
    assert (cx, cy) == (110.0, 120.0)
    masks = zm.masks((500, 500))
    assert masks["Display A"][120, 110] and not masks["Display A"][300, 300]


def test_zone_config_validation():
    base = {"zones": [{"name": "A", "polygon": [[0, 0], [10, 0], [10, 10]]}]}
    assert config_from_dict(base).zones[0].name == "A"
    bad = [
        ({"zones": [{"name": "A", "polygon": [[0, 0], [1, 1]]}]}, "at least 3"),
        ({"zones": [{"name": "A", "polygon": [[0, 0], [10, 0], [600, 10]]}]}, "inside"),
        ({"zones": [{"name": "A", "polygon": [[0, 0], [5, 5], [10, 10]]}]}, "zero area"),
        ({"zones": [{"name": "A", "polygon": SQ}, {"name": "A", "polygon": SQ}]}, "duplicate"),
        ({"zones": [{"name": "A", "polygon": SQ, "kind": "lobby"}]}, "kind"),
        ({"zones": [{"name": "", "polygon": SQ}]}, "name"),
        ({"zones": [{"name": "A", "polygon": SQ}], "analytics": {"funnel": ["B"]}}, "funnel"),
        ({"zones": [{"name": "A", "polygon": SQ, "colour": "red"}]}, "unknown"),
    ]
    for patch, msg in bad:
        with pytest.raises(ConfigError, match=msg):
            config_from_dict(patch)


def tracker(**kw):
    return SessionTracker(ZoneMap(sample_zones()), **{"min_zone_dwell_s": 2.0, "switch_hold_s": 0.5,
                                                      "timeout_s": 5.0, **kw})


def walk(tr, tid, points, t0=0.0, dt=0.1):
    """Feed one person's floor positions; returns finished sessions."""
    done = []
    for i, p in enumerate(points):
        done += tr.update(t0 + i * dt, [(tid, p)])
    return done


A, B, CK, ENT = (100, 120), (380, 100), (420, 340), (250, 450)


def test_a_stop_in_a_zone_becomes_a_visit_with_dwell():
    tr = tracker()
    walk(tr, 1, [ENT] * 10 + [A] * 50 + [CK] * 30)                      # 1 s entrance, 5 s A, 3 s checkout
    (rec,) = tr.flush()
    assert rec.path[-2:] == ["Display A", "Checkout"]
    a = next(v for v in rec.visits if v.zone == "Display A")
    assert a.dwell_s == pytest.approx(5.0, abs=0.7)


def test_walking_past_a_zone_is_not_a_visit():
    tr = tracker()
    walk(tr, 1, [ENT] * 30 + [A] * 8 + [ENT] * 30)                      # 0.8 s inside Display A
    (rec,) = tr.flush()
    assert "Display A" not in rec.path


def test_border_flicker_does_not_split_a_visit():
    tr = tracker()
    inside, outside = (100, 120), (100, 200)                            # just outside the zone
    pts = [inside] * 20 + [outside, inside, outside, inside] * 2 + [inside] * 20
    walk(tr, 1, pts)
    (rec,) = tr.flush()
    visits = [v for v in rec.visits if v.zone == "Display A"]
    assert len(visits) == 1 and visits[0].dwell_s > 3.5


def test_timeout_finishes_a_session_and_flags_are_recorded():
    tr = tracker(timeout_s=3.0)
    for i in range(30):
        tr.update(i * 0.1, [(7, A)], [(7, "IN")] if i == 5 else [])
    assert tr.update(5.0, [], []) == []                                  # within timeout
    (rec,) = tr.update(10.0, [], [])
    assert rec.entered and not rec.exited and rec.dwell_s == pytest.approx(2.9, abs=0.2)
    assert not tr.live


def test_two_visitors_are_independent_and_live_counts_work():
    tr = tracker()
    for i in range(40):
        tr.update(i * 0.1, [(1, A), (2, B)])
    counts = tr.current_zone_counts()
    assert counts["Display A"] == 1 and counts["Display B"] == 1 and counts["Checkout"] == 0
    recs = tr.flush()
    assert {tuple(r.path) for r in recs} == {("Display A",), ("Display B",)}


def test_first_sighting_inside_a_zone_counts_from_then():
    tr = tracker()
    walk(tr, 1, [A] * 40)
    (rec,) = tr.flush()
    assert rec.path == ["Display A"] and rec.visits[0].dwell_s == pytest.approx(3.9, abs=0.2)


def test_default_config_has_no_zones_until_added():
    assert default_config().zones == []


def test_stitching_continues_a_session_when_the_tracker_changes_the_id():
    tr = tracker(stitch_s=3.0)
    for i in range(40):
        tr.update(i * 0.1, [(1, A)])                                       # id 1 stands in Display A for 4 s
    for i in range(40, 80):
        tr.update(i * 0.1, [(2, (105, 124))])                              # id 2 appears right there; id 1 vanished
    (rec,) = tr.flush()
    assert rec.path == ["Display A"] and rec.visits[0].dwell_s == pytest.approx(7.9, abs=0.3)


def test_stitching_does_not_take_over_hidden_or_still_observed_sessions_or_far_ones():
    tr = tracker()
    for i in range(20):
        tr.update(i * 0.1, [(1, A), (2, B)])
    tr.update(2.0, [(3, (110, 125))], alive=[1])                           # id 1 is hidden, not lost
    assert 1 in tr.live and 3 in tr.live and len(tr.live) == 3
    tr2 = tracker()
    for i in range(20):
        tr2.update(i * 0.1, [(1, A)])
    tr2.update(2.2, [(9, (450, 40))])                                      # far away: a different person
    assert len(tr2.live) == 2
    tr3 = tracker(stitch_s=0.0)
    for i in range(20):
        tr3.update(i * 0.1, [(1, A)])
    tr3.update(2.2, [(9, A)])
    assert len(tr3.live) == 2                                              # stitching switched off


def test_no_stitching_at_the_entrance():
    tr = tracker()
    for i in range(20):
        tr.update(i * 0.1, [(1, ENT)])
    tr.update(2.2, [(2, (260, 455))])                                      # a new arrival right where someone left
    assert len(tr.live) == 2
