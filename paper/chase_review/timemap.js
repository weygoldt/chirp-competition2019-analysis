// Video frame <-> recording time with the LED map of a recording (led_alignment.LedAlignment).
// Used by index.html; tested against the Python map in tests/test_chase_review.py (node).
"use strict";

// searchsorted(bounds, f, side="right"): the number of bounds <= f
function segmentOf(map, f) {
  let lo = 0, hi = map.bounds.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (map.bounds[mid] <= f) lo = mid + 1; else hi = mid; }
  return lo;
}

// recording time (s) of video frame f
function toRec(map, f) { const k = segmentOf(map, f); return map.slope[k] * f + map.icpt[k]; }

// the frame in [lo, hi] whose recording time is nearest to t (the map increases with f)
function toFrame(map, t, lo, hi) {
  let a = lo, b = hi;
  while (a < b) { const mid = (a + b) >> 1; if (toRec(map, mid) < t) a = mid + 1; else b = mid; }
  if (a > lo && Math.abs(toRec(map, a - 1) - t) <= Math.abs(toRec(map, a) - t)) return a - 1;
  return a;
}

// the clip window of a chase: from preS before its onset to postS after its offset (recording s),
// clamped to the video; returns {frame0, n}
function chaseWindow(map, onFrame, offFrame, preS, postS, nFrames) {
  const f0 = toFrame(map, toRec(map, onFrame) - preS, 0, nFrames - 1);
  const f1 = toFrame(map, toRec(map, offFrame) + postS, 0, nFrames - 1);
  return {frame0: Math.min(f0, onFrame), n: Math.max(f1, offFrame) - Math.min(f0, onFrame) + 1};
}

if (typeof module !== "undefined") module.exports = {segmentOf, toRec, toFrame, chaseWindow};
