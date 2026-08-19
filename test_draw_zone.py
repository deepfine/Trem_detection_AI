import pytest
import numpy as np

from draw_zone import html_editor, zone_data, zones_data


def test_zone_data_lines():
    assert zone_data("lines", [(1, 2), (3, 4), (5, 6), (7, 8)]) == {
        "lines": [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
    }


def test_zone_data_mask():
    assert zone_data("mask", [(1, 2), (3, 4), (5, 6)]) == {"mask": [(1, 2), (3, 4), (5, 6)]}


def test_zone_data_rejects_too_few_points():
    with pytest.raises(ValueError):
        zone_data("lines", [(1, 2)])


def test_html_editor_embeds_frame():
    html = html_editor(np.zeros((4, 4, 3), dtype=np.uint8), 2, "zones")
    assert "data:image/jpeg;base64," in html
    assert "Frame 2" in html
    assert "Add zone" in html


def test_zones_data():
    assert zones_data([{"name": "a", "level": "danger", "points": [[0, 0], [1, 0], [1, 1]]}]) == {
        "zones": [{"name": "a", "level": "danger", "points": [[0, 0], [1, 0], [1, 1]]}]
    }
