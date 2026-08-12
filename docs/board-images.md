# Board images and their calibration

The app draws problems on photographs of the two layouts rather than on an abstract grid. That
only works if normalized `(x, y)` coordinates land exactly on the holds, so the mapping from
coordinate to pixel is fitted, not eyeballed.

## The assets

`web/public/board/mirror.jpg` and `spray.jpg`, with `calibration.json` beside them. They are
committed: they are source assets, not something the export produces.

The originals are 4319x4320 layout renders. Each is cropped to the hold field plus a margin —
which also removes the "12X12 TB2 MIRROR LAYOUT" title — and scaled to 1400px on the long side,
JPEG quality 75, about 380 KB each.

## How the calibration was fitted

The board is a regular lattice: 33 columns and 35 rows of holds at 4-unit spacing, with the
outermost holds at exactly `x = 0`, `x = 1`, `y = 0`, and `y = 1`. So four numbers place the
whole grid: where `x = 0` and `x = 1` fall, and the same for `y`.

Fitting them:

1. Mark every pixel that is part of a hold rather than the grey wall — either coloured
   (`max - min` across channels above 18) or dark (luminance below 70).
2. Take a first guess from the extent of that mask, inset by half a grid step.
3. Hill-climb the four anchors to maximize how much hold-mask sits under the 498 projected
   positions, from a 24px step down to 0.25px.

Two things say the fit is right rather than merely plausible. The horizontal and vertical
periods come out at 108.08 and 107.97 pixels — a square lattice, which is what a 4-unit grid in
both directions has to be, and nothing in the fit forced that. And the projected crosses sit on
the holds in all four corners, where any error would show up worst.

The two layouts are framed slightly differently — about 13px apart horizontally — so each is
calibrated separately.

`calibration.json` stores the anchors as fractions of the image, so the app never depends on
the pixel size and the images can be re-scaled without refitting.

## Redoing it

The fitting scripts were throwaway. If the images are ever replaced, the method above is the
recipe; the check that matters is that the two periods agree and that the corners line up.

## Drawing

`BoardView` renders the photo, then a ring per selected hold — no fill, so the hold underneath
stays visible, which is the whole reason for using a photo. A dark halo sits under each ring so
it reads against cream holds as well as black ones.

The ring is deliberately wider than the grid spacing (0.68 of a step in radius) so it
encircles the hold rather than sitting on top of it. Rings of two adjacent selections then
overlap slightly, which reads fine. The click target is a separate, smaller circle at 0.46 of a
step: under half the spacing, so neighbouring targets never overlap and a click is never
ambiguous. Unselected positions are nothing but that invisible target.

Both radii and the stroke width scale with the grid step, so the markers hold their proportions
at any display size.
