import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from ..vector.graph import geojson_to_graph
from ..utils.cli import _func_wrapper
from itertools import repeat


def main():

    parser = argparse.ArgumentParser(
        description='Create training pixel masks from vector data',
        argument_default=None)

    parser.add_argument('--source_file', '-s', type=str,
                        help='Full path to file to create graph from.')
    parser.add_argument('--output_path', '-o', type=str,
                        help='Full path to the output file for the graph' +
                        ' object.')
    parser.add_argument('--road_type_field', '-r', type=str,
                        help='The name of the column in --source_file that' +
                        ' defines the road type of each linestring.')
    parser.add_argument('--first_edge_idx', '-e', type=int, default=0,
                        help='The numeric index to use for the first edge in' +
                        ' the graph. Defaults to 0.')
    parser.add_argument('--first_node_idx', '-n', type=int, default=0,
                        help='The numeric index to use for the first node in' +
                        ' the graph. Defaults to 0.')
    parser.add_argument('--weight_norm_field', '-wn', type=str,
                        help='The name of a column in --source_file to' +
                        ' weight edges with. If not provided, edge weights' +
                        ' are determined only by Euclidean distance. If' +
                        ' provided, edge weights are distance*weight.')
    parser.add_argument('--batch', '-b', action='store_true', default=False,
                        help='Use this flag if you wish to operate on' +
                        ' multiple files in batch. In this case,' +
                        ' --argument-csv must be provided. See help' +
                        ' for --argument_csv and the codebase docs at' +
                        ' https://cw-geodata.readthedocs.io for more info.')
    parser.add_argument('--argument_csv', '-a', type=str,
                        help='The reference file for variable values for' +
                        ' batch processing. It must contain columns to pass' +
                        ' the source_file and reference_image arguments, and' +
                        ' can additionally contain columns providing the' +
                        ' footprint_column and decimal_precision arguments' +
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

    if args.batch and args.argument_csv is None:
        raise ValueError(
            'To perform batch processing, you must provide both --batch and' +
            ' --argument_csv.')
    if args.argument_csv is None and args.source_file is None:
        raise ValueError(
            'You must provide a source file using either --source_file or' +
            ' --argument_csv.')

    if args.argument_csv is not None:
        arg_df = pd.read_csv(args.argument_csv)
    else:
        arg_df = pd.DataFrame({})
    if args.batch:
        if args.source_file is not None:
            arg_df['source_file'] = args.source_file
        if args.output_path is not None:
            arg_df['output_path'] = args.output_path
        if args.road_type_field is not None:
            arg_df['road_type_field'] = args.road_type_field
        arg_df['first_node_idx'] = args.first_node_idx
        arg_df['first_edge_idx'] = args.first_edge_idx
        if args.weight_norm_field is not None:
            arg_df['weight_norm_field'] = args.weight_norm_field
    else:
        arg_df['source_file'] = [args.source_file]
        arg_df['output_path'] = [args.output_path]
        arg_df['road_type_field'] = [args.road_type_field]
        arg_df['first_node_idx'] = [args.first_node_idx]
        arg_df['first_edge_idx'] = [args.first_edge_idx]
        arg_df['weight_norm_field'] = [args.weight_norm_field]

    arg_df = arg_df.rename(columns={'source_file': 'geojson',
                                    'first_edge_idx': 'edge_idx'})
    arg_dict_list = arg_df[['geojson', 'output_path', 'road_type_field',
                            'weight_norm_field', 'edge_idx', 'first_node_idx',
                            'output_path']
                           ].to_dict(orient='records')
    if not args.batch:
        result = geojson_to_graph(**arg_dict_list[0])
        if not args.output_path:
            return result

    else:
        with Pool(processes=args.workers) as pool:
            result = tqdm(pool.starmap(_func_wrapper,
                                       zip(repeat(geojson_to_graph),
                                           arg_dict_list)))
            pool.close()


if __name__ == '__main__':
    main()
