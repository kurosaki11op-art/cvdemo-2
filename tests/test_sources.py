import cv2
import numpy as np
import pytest

from nexus.sources import FrameSource, SourceError


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "c.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(5):
        vw.write(np.full((48, 64, 3), 50 + i * 10, np.uint8))
    vw.release()
    return str(path)


def test_file_reads_all_frames_then_finishes(clip):
    src = FrameSource(clip).open()
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    assert len(frames) == 5 and src.finished and src.is_file and not src.is_live
    assert src.fps == pytest.approx(10, abs=1)
    src.release()


def test_loop_wraps_and_reports_it_once(clip):
    src = FrameSource(clip, loop=True).open()
    for _ in range(5):
        assert src.read() is not None
    assert not src.consume_wrap()
    assert src.read() is not None
    assert src.consume_wrap() and not src.consume_wrap()
    assert not src.finished


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        FrameSource(str(tmp_path / "nope.mp4")).open()


def test_source_kinds():
    assert FrameSource("0").is_webcam and FrameSource("2").source == 2
    s = FrameSource("rtsp://cam/stream")
    assert s.is_stream and s.is_live and not s.is_file
    assert FrameSource("clip.mp4").is_file
