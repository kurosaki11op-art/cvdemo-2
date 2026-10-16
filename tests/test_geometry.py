import numpy as np
import pytest

from nexus.geometry import FloorMapper, TripwireCounter

A, B = (100, 200), (500, 200)          # horizontal line, left -> right


def feed(tc, tid, points):
    return [tc.update(tid, p) for p in points]


def test_down_is_in_up_is_out():
    tc = TripwireCounter(A, B)
    assert "IN" in feed(tc, 1, [(300, 150), (300, 190), (300, 215)])
    assert "OUT" in feed(tc, 2, [(300, 250), (300, 210), (300, 185)])
    assert (tc.entries, tc.exits) == (1, 1)


def test_right_to_left_swaps_directions():
    tc = TripwireCounter(A, B, in_direction="right_to_left")
    feed(tc, 1, [(300, 150), (300, 190), (300, 215)])
    assert (tc.entries, tc.exits) == (0, 1)


def test_crossing_outside_the_segment_is_ignored():
    tc = TripwireCounter(A, B)
    feed(tc, 1, [(900, 150), (900, 250)])
    assert (tc.entries, tc.exits) == (0, 0)


def test_jitter_inside_deadband_is_ignored():
    tc = TripwireCounter(A, B, deadband_px=6)
    feed(tc, 1, [(300, 150), (300, 197), (300, 203), (300, 198), (300, 202), (300, 150)])
    assert (tc.entries, tc.exits) == (0, 0)


def test_each_crossing_counts_once_and_rearms_after_moving_away():
    tc = TripwireCounter(A, B, deadband_px=6, rearm_px=40)
    feed(tc, 1, [(300, 150), (300, 215)])               # IN
    feed(tc, 1, [(300, 185)])                           # back over the line, but not re-armed
    assert (tc.entries, tc.exits) == (1, 0)
    feed(tc, 1, [(300, 260), (300, 185)])               # moved well away, then returns: OUT
    assert (tc.entries, tc.exits) == (1, 1)


def test_new_track_first_seen_on_a_side_does_not_count():
    tc = TripwireCounter(A, B)
    assert feed(tc, 1, [(300, 250), (300, 260)]) == [None, None]


def test_forget_and_reset():
    tc = TripwireCounter(A, B)
    feed(tc, 1, [(300, 150), (300, 250)])
    tc.forget([])
    assert not tc.prev_side and not tc.counted_ids
    tc.reset()
    assert (tc.entries, tc.exits) == (0, 0)


def test_degenerate_line_and_direction_rejected():
    with pytest.raises(ValueError):
        TripwireCounter((1, 1), (1, 1))
    with pytest.raises(ValueError):
        TripwireCounter(A, B, in_direction="sideways")


def test_into_vector_points_to_in_side():
    tc = TripwireCounter(A, B)                            # IN = down the screen
    assert tc.into_vector[1] > 0.99
    assert TripwireCounter(A, B, "right_to_left").into_vector[1] < -0.99


def test_floor_mapper_maps_corners_and_handles_empty():
    src = [(190, 167), (450, 167), (590, 467), (50, 467)]
    dst = [(0, 0), (499, 0), (499, 499), (0, 499)]
    m = FloorMapper(src, dst)
    assert np.allclose(m.transform(src), dst, atol=1e-2)
    assert m.transform([]).shape == (0, 2)
    with pytest.raises(ValueError):
        FloorMapper(src[:3], dst)


def test_wobble_after_a_count_does_not_double_count():
    tc = TripwireCounter(A, B, deadband_px=6, rearm_px=40)
    feed(tc, 1, [(300, 150), (300, 215), (300, 185), (300, 215), (300, 190), (300, 230), (300, 250)])
    assert (tc.entries, tc.exits) == (1, 0)
