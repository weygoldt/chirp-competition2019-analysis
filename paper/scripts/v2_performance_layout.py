"""Layout of Fig. 10 (``plot_v2_performance.py``) in centimetres.

Every position is an explicit cm offset (author, 2026-11-25: rows span the
same width, axes of a row share their top and bottom edges, panels of one
kind are equally wide, label and letter space is reserved explicitly, and
groups are separated by visibly more whitespace than panels within a group):

- one row per pipeline stage: detection (A-D), assignment (E-H), chirp
  parameters (I-K); every row's leftmost axes start at ``LEFT`` and its
  rightmost axes end at ``RIGHT``;
- detection: four equally wide panels, simulated (A, B) and real recordings
  (C, D) separated by the wider ``GROUP_GAP_1``; each pair under a group
  heading on a thin rule that spans the pair from its first letter to its
  last axes (the text starts at the pair's first axes, so that no panel
  letter sits under it); C and D share the y-axis;
- assignment: four equally wide panels in two pairs that each share a
  y-axis, E-F (ΔF) and G-H (true chirp size), separated by ``GROUP_GAP_2``;
- chirp parameters: three equal squares, evenly spaced;
- every row under a group heading (author, 2026-11-25: per-row headings, so
  that E-K, all simulated, do not read as continuing the columns of C, D):
  "Simulated" and "Real recordings" over the detection row, "Simulated" over
  the assignment and chirp-parameter rows, each ``HEADING_DY`` above its row;
- between the rows ``ROW_GAP`` of whitespace from the lowest x-axis label of
  one row to the top of the next row's heading.

Gaps between panels reserve the right panel's y tick labels and y-axis label
(``YLABEL_1``: one label line, ``YLABEL_2``: two), plus visible whitespace.
Pure geometry (no data, no matplotlib), so that it can be tested.
"""

from __future__ import annotations

from dataclasses import dataclass

W, H = 17.4, 15.05  # figure [cm]: full text width; at most 15.05 cm tall (caption fits the page)

# every row: left edge of its leftmost, right edge of its rightmost axes (left: room
# for I's y-axis label beside its 4-digit tick labels)
LEFT, RIGHT = 1.62, 17.3
SHARED_GAP = 0.55  # between two panels sharing a y-axis (no y tick labels on the right one)
YLABEL_1 = 1.3  # y tick labels ("0.85") and a one-line y-axis label
YLABEL_2 = 1.65  # y tick labels and a two-line y-axis label
SPACE = 0.35  # visible whitespace between a panel and the next one's y-axis label
GROUP_EXTRA = 0.75  # extra whitespace between two groups of panels within a row
GAP_1 = YLABEL_1 + SPACE  # A -> B
GROUP_GAP_1 = YLABEL_2 + SPACE + GROUP_EXTRA  # B -> C: simulated | real recordings
GROUP_GAP_2 = YLABEL_1 + SPACE + GROUP_EXTRA  # F -> G: by ΔF | by true chirp size

XLABEL = 1.1  # below an axes: spine offset, x tick labels and the x-axis label
LETTER_DY = 0.3  # panel letters: centre this far above the axes' top
LETTER_HALF = 0.14  # half the height of a panel letter
ROW_GAP = 0.45  # whitespace between one row's x-axis labels and the next row's heading

KEY_Y = 14.8  # centre of the figure key (detector, expert)
TOP_1 = 13.65  # top of the detection row
HEADING_DY = 0.67  # centre of a row's group heading above the row's top
HEADING_HALF = 0.14  # half the height of a heading
HEADING_Y = TOP_1 + HEADING_DY  # centre of the group headings of the detection row
PANEL_H = 2.3  # height of the detection and assignment panels
SQUARE = 3.1  # side of the chirp-parameter panels
BOTTOM_MARGIN = 0.1  # below the lowest x-axis label


@dataclass(frozen=True)
class Box:
    """An axes: lower left corner (x, y), width and height [cm]."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h


def row(names: list[str], gaps: list[float], y: float, h: float) -> dict[str, Box]:
    """Equally wide panels from ``LEFT`` to ``RIGHT`` with the given gaps between them."""
    if len(gaps) != len(names) - 1:
        raise ValueError("one gap between each two panels")
    w = (RIGHT - LEFT - sum(gaps)) / len(names)
    out, x = {}, LEFT
    for i, name in enumerate(names):
        out[name] = Box(x, y, w, h)
        x += w + (gaps[i] if i < len(gaps) else 0.0)
    return out


def next_top(bottom: float) -> float:
    """Top of the row below a row whose axes end at ``bottom``."""
    return bottom - XLABEL - ROW_GAP - (HEADING_DY + HEADING_HALF)


def heading_y(top: float) -> float:
    """Centre of the group heading of the row whose axes top is ``top``."""
    return top + HEADING_DY


def layout() -> dict[str, Box]:
    """The axes of Fig. 10, A-K."""
    top_2 = next_top(TOP_1 - PANEL_H)
    top_3 = next_top(top_2 - PANEL_H)
    boxes = {
        **row(list("ABCD"), [GAP_1, GROUP_GAP_1, SHARED_GAP], TOP_1 - PANEL_H, PANEL_H),
        **row(list("EFGH"), [SHARED_GAP, GROUP_GAP_2, SHARED_GAP], top_2 - PANEL_H, PANEL_H),
    }
    gap_3 = (RIGHT - LEFT - 3 * SQUARE) / 2
    boxes.update(row(list("IJK"), [gap_3, gap_3], top_3 - SQUARE, SQUARE))
    if boxes["I"].y - XLABEL < BOTTOM_MARGIN:
        raise ValueError(f"the chirp-parameter row does not fit: bottom {boxes['I'].y:.2f} cm")
    if gap_3 < YLABEL_2 + SPACE:
        raise ValueError(f"no room for the y-axis labels of J and K: gap {gap_3:.2f} cm")
    return boxes


ROWS = ("ABCD", "EFGH", "IJK")
SHARED = "DFH"  # panels that share their left neighbour's y-axis (no y tick labels)
TWO_LINE = "CK"  # panels with a two-line y-axis label


def letters() -> dict[str, float]:
    """Distance of each panel letter's centre left of its axes [cm]: beyond the
    y-axis label, or just left of a shared axis; so A, E and I share one x."""
    out = {}
    for name in "".join(ROWS):
        if name in SHARED:
            out[name] = SHARED_GAP - 0.2
        elif name in TWO_LINE:
            out[name] = YLABEL_2
        else:
            out[name] = YLABEL_1
    return out


def extent_table(boxes: dict[str, Box]) -> str:
    """The axes' extents [cm] as a fixed-width table (left, right, bottom, top, width, height)."""
    lines = [f"{'axes':<5}{'left':>7}{'right':>7}{'bottom':>8}{'top':>7}{'width':>7}{'height':>7}"]
    for r in ROWS:
        for name in r:
            b = boxes[name]
            lines.append(f"{name:<5}{b.x:7.3f}{b.right:7.3f}{b.y:8.3f}{b.top:7.3f}"
                         f"{b.w:7.3f}{b.h:7.3f}")
    return "\n".join(lines)
