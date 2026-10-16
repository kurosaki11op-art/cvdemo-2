import numpy as np

from nexus.identity import IdentityFilter, contained_duplicates


def scene(*people):
    """Frame with coloured rectangles: people = (x, colour_bgr)."""
    f = np.full((480, 640, 3), 40, np.uint8)
    boxes = []
    for x, col in people:
        f[100:400, x:x + 120] = col
        boxes.append([x, 100, x + 120, 400])
    return f, np.array(boxes, float)


RED, GREEN = (30, 30, 220), (40, 200, 40)


def test_contained_box_is_dropped():
    xyxy = np.array([[100, 100, 300, 400], [140, 150, 240, 350.]])
    assert contained_duplicates(xyxy, np.array([0.9, 0.5])) == {1}
    assert contained_duplicates(xyxy[:1], np.array([0.9])) == set()
    far = np.array([[0, 0, 50, 50], [200, 200, 300, 300.]])
    assert contained_duplicates(far, np.array([0.9, 0.9])) == set()


def test_track_needs_min_frames():
    idf = IdentityFilter(min_frames=3)
    f, b = scene((100, RED))
    shown = [bool(idf.update(f, b, [1], np.array([0.9]))[0]) for _ in range(4)]
    assert shown == [False, False, True, True]


def test_id_switch_is_repaired():
    idf = IdentityFilter(min_frames=2)
    f, b = scene((100, RED))
    for _ in range(3):
        idf.update(f, b, [1], np.array([0.9]))
    f2, b2 = scene((110, RED))
    _, ids = idf.update(f2, b2, [7], np.array([0.9]))       # tracker re-issued the id
    assert ids == [1]


def test_id_switch_with_different_look_is_a_new_person():
    idf = IdentityFilter(min_frames=1)
    f, b = scene((100, RED))
    idf.update(f, b, [1], np.array([0.9]))
    f2, b2 = scene((110, GREEN))
    _, ids = idf.update(f2, b2, [7], np.array([0.9]))
    assert ids == [7]


def test_reid_can_be_disabled():
    idf = IdentityFilter(min_frames=1, reid=False)
    f, b = scene((100, RED))
    idf.update(f, b, [1], np.array([0.9]))
    f2, b2 = scene((110, RED))
    assert idf.update(f2, b2, [7], np.array([0.9]))[1] == [7]


def test_twin_with_same_colours_is_suppressed_but_different_people_stay():
    f, b = scene((50, RED), (400, RED))
    kept, _ = IdentityFilter(min_frames=1).update(f, b, [1, 2], np.array([0.9, 0.8]))
    assert kept == [0]                                       # mirror image removed
    kept, _ = IdentityFilter(min_frames=1, twin=False).update(f, b, [1, 2], np.array([0.9, 0.8]))
    assert kept == [0, 1]
    f, b = scene((50, RED), (400, GREEN))
    kept, _ = IdentityFilter(min_frames=1).update(f, b, [1, 2], np.array([0.9, 0.8]))
    assert kept == [0, 1]


def test_old_identities_are_pruned():
    idf = IdentityFilter(min_frames=1, forget_frames=10)
    f, b = scene((100, RED))
    idf.update(f, b, [1], np.array([0.9]))
    empty = np.empty((0, 4))
    for _ in range(300):
        idf.update(f, empty, [], np.array([]))
    assert 1 not in idf.active_ids() and not idf.canon


def test_small_reflection_is_merged_and_the_more_confident_box_wins():
    f = np.full((480, 640, 3), 40, np.uint8)
    f[100:400, 50:170] = RED          # the person: large
    f[200:300, 400:450] = RED         # the reflection: same colours, about 1/7 of the area
    boxes = np.array([[50, 100, 170, 400], [400, 200, 450, 300.]])
    kept, _ = IdentityFilter(min_frames=1).update(f, boxes, [1, 2], np.array([0.55, 0.9]))
    assert kept == [1]                                       # the more confident box survives


def test_an_older_but_fainter_track_does_not_beat_the_real_person():
    idf = IdentityFilter(min_frames=1)
    f, b = scene((400, RED))
    for _ in range(8):                                       # the faint ghost is established first
        idf.update(f, b, [2], np.array([0.5]))
    f2 = np.full((480, 640, 3), 40, np.uint8)
    f2[100:400, 50:170] = RED
    f2[100:400, 400:520] = RED
    both = np.array([[50, 100, 170, 400], [400, 100, 520, 400.]])
    kept, ids = idf.update(f2, both, [1, 2], np.array([0.9, 0.5]))
    assert ids == [1] and kept == [0]
