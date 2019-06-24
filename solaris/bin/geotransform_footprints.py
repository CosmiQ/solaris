import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from ..vector.polygon import geojson_to_px_gdf
from ..vector.polygon import georegister_px_df
from ..utils.cli import _func_wrapper
from itertools import repeat


def main():

    parser = argparse.ArgumentParser(
        description='Interconvert footprints between pixel and geographic ' +
        'coordinate systems.', argument_default=None)

    parser.add_argument('--source_file', '-s', type=str,
                        help='Full path to file to transform')
    parser.add_argument('--reference_image', '-r', type=str,
                        help='Full path to a georegistered image in the same' +
                        ' coordinate system (for conversion to pixels) or in' +
                        ' the target coordinate system (for conversion to a' +
                        ' geographic coordinate reference system).')
    parser.add_argument('--output_path', '-o', type=str,
                        help='Full path to the output file for the converted' +
                        'footprints.')
    parser.add_argument('--to_pixel', '-p', action='store_true', default=False,
                        help='Use this argument if you wish to convert' +
                        ' footprints in --source-file to pixel coordinates.')
    parser.add_argument('--to_geo', '-g', action='store_true', default=False,
                        help='Use this argument if you wish to convert' +
                        ' footprints in --source-file to a geographic' +
                        ' coordinate system.')
    parser.add_argument('--geometry_column', '-c', type=str,
                        default='geometry', help='The column containing' +
                        ' footprint polygons to transform. If not provided,' +
                        ' defaults to "geometry".')
    parser.add_argument('--decimal_precision', '-d', type=int,
                        help='The number of decimals to round to in the' +
                        ' final footprint geometries. If not provided, they' +
                        ' will be rounded to float32 precision.')
    parser.add_argument('--batch', '-b', action='store_true', default=False,
                        help='Use this flag if you wish to operate on' +
                        ' multiple files in batch. In this case,' +
                        ' --argument-csv must be provided. See help' +
                        ' for --argument_csv and the codebase docs at' +
                        ' https://solaris.readthedocs.io for more info.')
    parser.add_argument('--argument_csv', '-a', type=str,
                        help='The reference file for variable values for' +
                        ' batch processing. It must contain columns to pass' +
                        ' the source_file and reference_image arguments, and' +
                        ' can additionally contain columns providing the' +
                        ' geometry_column and decimal_precision arguments' +
                        ' if you wish to define them differently for items' +
                        ' in the batch. These columns must have the same' +
                        ' names as the corresponding arguments. See the ' +
                        ' usage recipes at https://cw-geodata.readthedocs.io' +
                        ' for examples.')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='The number of parallel processing workers to' +
                        ' use. This should not exceed the number of CPU' +
                        ' cores available.')

    args = parser.parse_args()
    # check that the necessary set of arguments are provided.
    if args.batch and args.argument_csv is None:
        raise ValueError(
            'To perform batch processing, you must provide both --batch and' +
            ' --argument_csv.')
    if args.argument_csv is None and args.source_file is None:
        raise ValueError(
            'You must provide a source file using either --source_file or' +
            ' --argument_csv.')
    if args.argument_csv is None and args.reference_image is None:
        raise ValueError(
            'You must provide a reference image using either' +
            ' --reference_image or --argument_csv.')
    if args.to_pixel == args.to_geo:
        raise ValueError(
            'One, and only one, of --to_pixel and --to_geo must be specified.')

    if args.argument_csv is not None:
        arg_df = pd.read_csv(args.argument_csv)
    else:
        arg_df = pd.DataFrame({})

    if args.batch:
        # add values from individual arguments to the argument df
        if args.source_file is not None:
            arg_df['source_file'] = args.source_file
        if args.reference_image is not None:
            arg_df['reference_image'] = args.reference_image
        if args.geometry_column is not None:
            arg_df['geometry_column'] = args.geometry_column
        if args.decimal_precision is not None:
            arg_df['decimal_precision'] = args.decimal_precision
    else:
        # add values from individual arguments to the argument df
        if args.source_file is not None:
            arg_df['source_file'] = [args.source_file]
        if args.reference_image is not None:
            arg_df['reference_image'] = [args.reference_image]
        if args.geometry_column is not None:
            arg_df['geometry_column'] = [args.geometry_column]
        if args.decimal_precision is not None:
            arg_df['decimal_precision'] = [args.decimal_precision]
        if args.output_path is not None:
            arg_df['output_path'] = [args.output_path]

    if args.to_pixel:
        # rename argument columns for compatibility with the target func
        arg_df = arg_df.rename(columns={'source_file': 'geojson',
                                        'reference_image': 'im_path',
                                        'decimal_precision': 'precision',
                                        'geometry_column': 'geom_col'})
        arg_dict_list = arg_df[
            ['geojson', 'im_path', 'precision', 'geom_col', 'output_path']
            ].to_dict(orient='records')
        func_to_call = geojson_to_px_gdf
    elif args.to_geo:
        # rename argument columns for compatibility with the target func
        arg_df = arg_df.rename(columns={'source_file': 'df',
                                        'reference_image': 'im_path',
                                        'decimal_precision': 'precision',
                                        'geometry_column': 'geom_col'})
        arg_dict_list = arg_df[
            ['df', 'im_path', 'precision', 'geom_col', 'output_path']
            ].to_dict(orient='records')
        func_to_call = georegister_px_df

    if not args.batch:
        result = func_to_call(**arg_dict_list[0])
        if not args.output_path:
            return result
    else:
        with Pool(processes=args.workers) as pool:
            result = tqdm(pool.starmap(_func_wrapper, zip(repeat(func_to_call),
                                                          arg_dict_list)))
            pool.close()


if __name__ == '__main__':
    main()
