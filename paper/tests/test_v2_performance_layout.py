"""The layout of Fig. 10 (v2_performance_layout): the author's alignment rules."""

from __future__ import annotations

import pytest

import v2_performance_layout as L

TOL = 1e-9


@pytest.fixture(scope="module")
def boxes():
    return L.layout()


def test_every_row_spans_the_same_width(boxes):
    for row in L.ROWS:
        assert boxes[row[0]].x == pytest.approx(L.LEFT, abs=TOL)
        assert boxes[row[-1]].right == pytest.approx(L.RIGHT, abs=TOL)


def test_axes_of_a_row_share_top_and_bottom_and_width(boxes):
    for row in L.ROWS:
        first = boxes[row[0]]
        for name in row[1:]:
            b = boxes[name]
            assert (b.y, b.top, b.w) == pytest.approx((first.y, first.top, first.w), abs=TOL)


def test_chirp_parameter_panels_are_equal_squares_evenly_spaced(boxes):
    i, j, k = (boxes[n] for n in "IJK")
    assert i.w == pytest.approx(i.h, abs=TOL)
    assert j.x - i.right == pytest.approx(k.x - j.right, abs=TOL)


def test_groups_are_separated_by_more_whitespace_than_panels_within_them(boxes):
    a, b, c, d = (boxes[n] for n in "ABCD")
    e, f, g, h = (boxes[n] for n in "EFGH")
    # between groups: the next panel's label space plus extra whitespace
    assert c.x - b.right - L.YLABEL_2 > b.x - a.right - L.YLABEL_1
    assert g.x - f.right - L.YLABEL_1 > L.SPACE
    assert g.x - f.right > f.x - e.right
    assert d.x - c.right == pytest.approx(L.SHARED_GAP, abs=TOL)
    assert h.x - g.right == pytest.approx(L.SHARED_GAP, abs=TOL)


def test_rows_are_separated_by_row_gap_and_everything_fits_the_page(boxes):
    top_1, bottom_1 = boxes["A"].top, boxes["A"].y
    top_2, bottom_2 = boxes["E"].top, boxes["E"].y
    top_3, bottom_3 = boxes["I"].top, boxes["I"].y
    letter_top = L.LETTER_DY + L.LETTER_HALF
    heading_top = L.HEADING_DY + L.HEADING_HALF
    assert bottom_1 - L.XLABEL - (top_2 + heading_top) == pytest.approx(L.ROW_GAP, abs=TOL)
    assert bottom_2 - L.XLABEL - (top_3 + heading_top) == pytest.approx(L.ROW_GAP, abs=TOL)
    assert bottom_3 - L.XLABEL >= L.BOTTOM_MARGIN - TOL
    for top in (top_1, top_2, top_3):  # every row's heading sits above its letters
        assert L.heading_y(top) - L.HEADING_HALF > top + letter_top
    assert L.heading_y(top_1) == pytest.approx(L.HEADING_Y, abs=TOL)
    assert L.HEADING_Y + L.HEADING_HALF < L.KEY_Y < L.H
    assert L.W == 17.4
    assert L.H <= 15.05


def test_letters_of_the_first_column_share_one_x(boxes):
    dx = L.letters()
    xs = {boxes[n].x - dx[n] for n in "AEI"}
    assert len({round(x, 9) for x in xs}) == 1
    assert min(xs) > L.LETTER_HALF  # inside the page


def test_extent_table_lists_every_axes_once(boxes):
    lines = L.extent_table(boxes).splitlines()
    assert [ln.split()[0] for ln in lines[1:]] == list("ABCDEFGHIJK")
