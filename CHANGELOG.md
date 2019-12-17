# Changelog

## Instructions

Anytime you add something new to this project, add a new item under the appropriate sub-heading of the [Unreleased](#unreleased) portion of this document. That item should be formatted as follows:
```
- [Date (ISO format)], [GitHub username]: [Short description of change] [(PR number)]
```
e.g.
```
- 20190930, nrweir: Added changelog (#259)
```
Consistent with the "one PR per task" paradigm, we recommend having only one changelog entry per PR whenever possible; however, multiple entries can be included for a single PR if needed to capture the full changeset.

When a new version of `solaris` is released, all of the changes in the Unreleased portion will be moved to the newest release version.

## Unreleased

### Added
### Removed
### Changed
### Fixed
### Deprecated
### Security

## Version 0.2.0

### Added
- 20190930, nrweir: Added CHANGELOG.md (#259)
- 20190930, nrweir: Add contributing guidelines, CONTRIBUTING.md (#260)
- 20191003, nrweir: Added `solaris.vector.mask.instance_mask()` (#261)
- 20191009, nrweir: Added `solaris.data.coco` and some label utility functions (#265)
- 20191009, nrweir: Added `solaris.data.coco` API documentation and a usage tutorial (#266)
- 20191122, dphogan: Added option to take sigmoid of input in TorchDiceLoss (#281)
- 20191122, dphogan: Inferer calls now take default DataFrame path from config dictionary (#282)
- 20191125, nrweir: Added `solaris.utils.data.make_dataset_csv()` (#241)
- 20191202, dphogan: Added fixed nodata value of 0 for mask files (#295)
- 20191203: dphogan: Added filename argument to vector tiler's tile() (#297)
- 20191211: rbavery: Tilers also accept rasterio CRS objects, `RasterTiler.tile` returns CRS object for vector tiler (#294)
- 20191214: rbavery: tiler argument `aoi_bounds` is now `aoi_boundary` and can accept polygons besides boxes. functionaility for this moved to `solaris.utils.geo.split_geom` (#298)
- 20191217: dphogan: Added support for custom loss functions (#308)

### Fixed
- 20191123, dphogan: Fixed issue in mask_to_poly_geojson() with empty GeoDataFrames.
- 20191204, dphogan: Fixed issue with file output from footprint_mask() and contact_mask() (#301)
- 20191212, jshermeyer: Fixed issue with vector tiling: could not load in list of sublists previously. Corrected comments for appropriate order as well. (#306)


---
_The changelog for solaris was not implemented until after version 0.1.3, therefore no previous changes are recorded here. See the [GitHub releases](https://github.com/CosmiQ/solaris/releases) for available change records._
