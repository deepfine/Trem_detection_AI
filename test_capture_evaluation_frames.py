import pytest

from capture_evaluation_frames import capture, preserve_frame


def test_preserves_original_bytes_and_camera_identity(tmp_path):
    root = tmp_path / "live"
    source = root / "33_3/2026/10/27/a.jpg"
    source.parent.mkdir(parents=True)
    data = b"\xff\xd8test-frame\xff\xd9"
    source.write_bytes(data)
    output = tmp_path / "saved"
    record = preserve_frame(source, root, output)
    assert record["camera_id"] == 3
    assert record["bytes"] == len(data)
    assert (output / record["file"]).read_bytes() == source.read_bytes() == data
    assert record["stage"] == "before_analysis_overlay"
    with pytest.raises(FileExistsError):
        preserve_frame(source, root, output)


@pytest.mark.parametrize("data", [b"", b"not-a-jpeg", b"\xff\xd8unfinished"])
def test_does_not_preserve_incomplete_images(tmp_path, data):
    root = tmp_path / "live"
    source = root / "33_3/2026/10/27/a.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    assert preserve_frame(source, root, tmp_path / "saved") is None


def test_output_cannot_be_inside_production_capture_root(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        capture(tmp_path, tmp_path / "saved", 1)


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_capture_rejects_invalid_duration(tmp_path, seconds):
    root = tmp_path / "live"
    root.mkdir()
    with pytest.raises(ValueError, match="seconds"):
        capture(root, tmp_path / "saved", seconds)
