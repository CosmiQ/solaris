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

---

## Version 0.4.0

### Added
- 20200803, jshermeyer: Added mAP metric
- 20200701, dphogan: Added SpaceNet Change and Object Tracking (SCOT) metric
### Removed
### Changed
### Fixed
### Deprecated
### Security

---

## Version 0.3.0

### Added
- 20200701, dphogan: Added multimodal preprocessing subpackage (#360)
- 20200716, dphogan: Added three-part tutorial for preprocessing subpackage
### Fixed
- 20200630, dphogan: Remove edge case numeric values from loss function tests (#358)
- 20200706, dphogan: Added API reference entry for utils.data
- 20200706, roshanr11: fixed `checkpoint_frequency` parameter in YAML files to work as intended
- 20200706, roshanr11: tqdm progress bar fixed, follow-up on previous issue (#335)

---

## Version 0.2.2

### Changed
- 20200401, rbavery: new tiler method `fill_all_nodata` to fill nodata with `"mean"` or custom value (#328)
- 20200401, rbavery: option to ignore MultiPolygon and GeometryCollection types in `geojson2coco` since these cannot be converted to COCO.
- 20200401, rbavery: new function `solaris.vector.mask.geojsons_to_masks_and_fill_nodata`, which rasterized vector labels according to raster tile extents. Fills nodata areas in raster tile and corresponding rasterized label raster.
- 20200401, rbavery: new test/example of tiling and creating instance masks with nodata values filled in tile outputs

### Fixed
- 20200401, rbavery: restrict_to_aoi implemented, sets values outside aoi to nodata value (#327, #240)
- 20200401, rbavery: tqdm prints correctly in notebook and lab (if ipywidgets is enabled according to https://github.com/tqdm/tqdm/issues/394#issuecomment-384743637) (#335)
- 20200401, rbavery: fixed bug where aoi boundary was not intersected with src_img extent prior to tiling
- 20200401, rbavery/nrweir: adapted `_check_crs` to convert `pyproj.CRS` or pass through `rasterio.crs.CRS` class when rasterio crs object is required (for example, reprojecting in the tilers)
- 20200414, zaburo-ch: fixed `val_datagen` to point to the correct augmentation pipeline

---

## Version 0.2.1

### Changed
- 20200103, nrweir: Updated version pins for proj6 compatibility, also relaxed version pins for many dependencies (#321)
### Fixed
- 20200103, nrweir: Fixed various places where CRS wasn't passed correctly from rasterio CRS object (#319, #322)
- 20200103, nrweir: Fixed axis length check for axis ordering in sol.utils.raster.reorder_axes() (#318)

---

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
- 20191219: rbavery: In `solaris.utils.geo.split_geom`, tile bounds that fall within `aoi_boundary` but not `src_img` are not returned. `solaris.vector.mask.instance_mask` only rasterizes geojsons where `reference_im` has values (nodata pixels won't have corresponding labels) (#315)


---
_The changelog for solaris was not implemented until after version 0.1.3, therefore no previous changes are recorded here. See the [GitHub releases](https://github.com/CosmiQ/solaris/releases) for available change records._
