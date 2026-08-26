# Withdrawn 2D benchmark checkpoints

Nothing in this directory feeds a published number. It is kept so that the rows
the manuscript used to print can still be reconstructed and audited.

## Unsourced constants — Daganzo, Chien, Kwon–Golden–Wasil

`results_daganzo*.csv`, `results_chien*.csv`, `results_kwon*.csv`

The coefficients behind these rows reached this repository through the
literature review in Figliozzi (2008), not through the primary articles. All
three DOIs resolve to paywalled records with no open-access location, so the
primaries could not be read. Secondary renderings also disagree with one
another, so there is no way to choose between them without the originals.
The estimators are gone from `classical_region_estimators.ESTIMATORS`, from
every benchmark runner, and from every table in the paper. The manuscript still
surveys the three works as prior art; it prints no number for them.

## Our own construction — `Cavdar_region`

`results_cavdar_region*.csv`

This row fed Çavdar–Sokol the generator support `G^2` as the area term. Çavdar
defines `A` as the covering rectangle of the *nodes*; the source has no
sampling-region concept at all, so the variant was our invention rather than a
source-faithful reading. It was also the one configuration in which the
source's own Eq. (21) correction made the result worse (signed bias +7.71 →
+18.14).

## Superseded — `results_cavdar_PRE_REBUILD.csv`

Çavdar–Sokol as scored before the rebuild against the primary document
(Çavdar's 2014 Georgia Tech dissertation, Ch. 4): no Eq. (21) correction, and
an axis-aligned bounding box in place of the minimum-area enclosing rectangle.
Retained only to show what moved. The live row is `../results_cavdar.csv`.
