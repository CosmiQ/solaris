import geopandas as gpd
import scipy.optimize
import scipy.sparse

def match_footprints(grnd_df, prop_df,
                     threshold=0.25, base_reward=100.):
    """
    Optimal matching of ground truth footprints with proposal footprints
    (for a single timestep).
    Input dataframes should have "id" & "geometry" columns.
    """

    # Supplement IDs with indices (which run from zero
    # to one less than the number of unique IDs)
    grnd_id_set = set(grnd_df['id'])
    prop_id_set = set(prop_df['id'])
    grnd_id_to_index = {id: index for index, id in
                        enumerate(sorted(list(grnd_id_set)))}
    prop_id_to_index = {id: index for index, id in
                        enumerate(sorted(list(prop_id_set)))}
    grnd_index_to_id = {index: id for id, index in grnd_id_to_index.items()}
    prop_index_to_id = {index: id for id, index in prop_id_to_index.items()}
    grnd_df['index'] = grnd_df.id.apply(lambda id: grnd_id_to_index[id])
    prop_df['index'] = prop_df.id.apply(lambda id: prop_id_to_index[id])

    # Calculate IOU for all intersections, and the corresponding reward
    grnd_df['grnd_area'] = grnd_df.area
    prop_df['prop_area'] = prop_df.area
    if not (grnd_df.empty or prop_df.empty):
        intersect = gpd.overlay(grnd_df, prop_df)
    else:
        intersect = None
    if intersect is None or len(intersect) == 0:
        return [], [], len(grnd_df), len(prop_df), 0, len(prop_df), len(grnd_df), 0., grnd_id_set, prop_id_set
    intersect['inter_area'] = intersect.area
    intersect['iou'] = intersect['inter_area'] / (intersect['grnd_area']
        + intersect['prop_area'] - intersect['inter_area'])
    intersect['reward'] = intersect.apply(lambda row: (row.iou > threshold)
                                          * (base_reward + row.iou), axis=1)

    # Convert IOUs and rewards to 2D arrays (by way of sparse matrices)
    iou_matrix = scipy.sparse.coo_matrix((intersect.iou, (intersect.index_1, intersect.index_2)),
        shape=(len(grnd_df), len(prop_df)))
    iou_arr = iou_matrix.toarray()
    reward_matrix = scipy.sparse.coo_matrix((intersect.reward, (intersect.index_1, intersect.index_2)),
        shape=(len(grnd_df), len(prop_df)))
    reward_arr = reward_matrix.toarray()

    # Solve unbalanced linear assignment problem
    grnd_match, prop_match = scipy.optimize.linear_sum_assignment(reward_arr, maximize=True)
    iou_match = iou_arr[grnd_match, prop_match]

    # Remove matches that don't actually contribute to the total score
    grnd_match_pruned = grnd_match[iou_match>threshold]
    prop_match_pruned = prop_match[iou_match>threshold]
    iou_match_pruned = iou_match[iou_match>threshold]

    # Look up IDs for each match, and calculate descriptive statistics
    grnd_match_ids = [grnd_index_to_id[index] for index in grnd_match_pruned]
    prop_match_ids = [prop_index_to_id[index] for index in prop_match_pruned]
    num_grnd = len(grnd_df)
    num_prop = len(prop_df)
    tp = len(iou_match_pruned)
    fp = num_prop - tp
    fn = num_grnd - tp
    if 2*tp + fp + fn > 0:
        f1 = (2*tp) / (2*tp + fp + fn)
    else:
        f1 = 0

    return grnd_match_ids, prop_match_ids, num_grnd, num_prop, tp, fp, fn, f1, grnd_id_set, prop_id_set


def scot_one_aoi(grnd_df, prop_df, threshold=0.25, base_reward=100., beta=2.,
                stats=False, verbose=False):
    """
    SpaceNet Change and Object Tracking (SCOT) metric, for one AOI.
    Input dataframes should have "timestep", "id", & "geometry" columns.
    """

    # Get list of timesteps from ground truth and proposal dataframes
    grnd_timestep_set = set(grnd_df.timestep.drop_duplicates())
    prop_timestep_set = set(grnd_df.timestep.drop_duplicates())
    timesteps = sorted(list(grnd_timestep_set.union(prop_timestep_set)))

    # Loop through timesteps
    if verbose:
        print('Matching footprints')
    tp_net = 0
    fp_net = 0
    fn_net = 0
    num_grnd_net = 0
    num_prop_net = 0
    all_grnd_ids = []
    all_prop_ids = []
    change_tp_net = 0
    change_fp_net = 0
    change_fn_net = 0
    change_grnd_ids = set()
    change_prop_ids = set()
    for i, timestep in enumerate(timesteps):

        # Get just the data for this timestep
        grnd_df_one_timestep = grnd_df.loc[grnd_df.timestep == timestep].copy()
        prop_df_one_timestep = prop_df.loc[prop_df.timestep == timestep].copy()

        # Find footprint matches for this timestep
        grnd_ids, prop_ids, num_grnd, num_prop, tp, fp, fn, f1, grnd_id_set, prop_id_set = match_footprints(
            grnd_df_one_timestep, prop_df_one_timestep,
            threshold=threshold, base_reward=base_reward)

        # Collect aggregate statistics for tracking, and retain all match IDs
        tp_net += tp
        fp_net += fp
        fn_net += fn
        num_grnd_net += num_grnd
        num_prop_net += num_prop
        all_grnd_ids = grnd_ids + all_grnd_ids # newest first
        all_prop_ids = prop_ids + all_prop_ids # newest first
        if verbose:
            print('  %2i: F1 = %.4f' % (i + 1, f1))

        # Collect aggregate statistics for change detection
        if i > 0:
            # Find change detection TPs, FPs, and FNs among matched footprints
            new_grnd = [grnd_id not in change_grnd_ids for grnd_id in grnd_ids]
            new_prop = [prop_id not in change_prop_ids for prop_id in prop_ids]
            change_tp_list = [g and p for g, p in zip(new_grnd, new_prop)]
            change_fp_list = [p and not g for g, p in zip(new_grnd, new_prop)]
            change_fn_list = [g and not p for g, p in zip(new_grnd, new_prop)]
            change_tp_net += sum(change_tp_list)
            change_fp_net += sum(change_fp_list)
            change_fn_net += sum(change_fn_list)
            # Find change detection FPs and FNs among unmatched footprints
            unmatched_fp = prop_id_set.difference(prop_ids).difference(change_prop_ids)
            unmatched_fn = grnd_id_set.difference(grnd_ids).difference(change_grnd_ids)
            change_fp_net += len(unmatched_fp)
            change_fn_net += len(unmatched_fn)
        change_grnd_ids = change_grnd_ids.union(grnd_id_set)
        change_prop_ids = change_prop_ids.union(prop_id_set)

    # Identify which matches are mismatches
    # (i.e., inconsistent with previous timesteps)
    if verbose:
        print('Identifying mismatches')
    mm_net = 0
    for i in range(len(all_grnd_ids)):
        grnd_id = all_grnd_ids[i]
        prop_id = all_prop_ids[i]
        previous_grnd_ids = all_grnd_ids[i+1:]
        previous_prop_ids = all_prop_ids[i+1:]
        grnd_mismatch = grnd_id in previous_grnd_ids and previous_prop_ids[previous_grnd_ids.index(grnd_id)] != prop_id
        prop_mismatch = prop_id in previous_prop_ids and previous_grnd_ids[previous_prop_ids.index(prop_id)] != grnd_id
        mismatch = grnd_mismatch or prop_mismatch
        if mismatch:
            mm_net += 1

    # Compute and return score according to the metric
    track_tp_net = tp_net - mm_net
    track_fp_net = fp_net + mm_net
    track_fn_net = fn_net + mm_net
    if track_tp_net + (track_fp_net + track_fn_net)/2. > 0:
        track_score = (track_tp_net) / (track_tp_net
                                        + (track_fp_net + track_fn_net)/2.)
    else:
        track_score = 0
    if change_tp_net + (change_fp_net + change_fn_net)/2. > 0:
        change_score = (change_tp_net) / (change_tp_net
                                          + (change_fp_net + change_fn_net)/2.)
    else:
        change_score = 0
    if beta * beta * change_score + track_score > 0:
        combo_score = (1 + beta * beta) * (change_score * track_score) / (beta * beta * change_score + track_score)
    else:
        combo_score = 0
    if verbose:
        print('Tracking:')
        print('    Mismatches: %i' % mm_net)
        print('      True Pos: %i' % track_tp_net)
        print('     False Pos: %i' % track_fp_net)
        print('     False Neg: %i' % track_fn_net)
        print('   Track Score: %.4f' % track_score)
        print('Change Detection:')
        print('      True Pos: %i' % change_tp_net)
        print('     False Pos: %i' % change_fp_net)
        print('     False Neg: %i' % change_fn_net)
        print('  Change Score: %.4f' % change_score)
        print('Combined Score: %.4f' % combo_score)
    if stats:
        return combo_score, [mm_net, track_tp_net, track_fp_net, track_fn_net,
                             track_score, change_tp_net, change_fp_net,
                             change_fn_net, change_score, combo_score]
    else:
        return combo_score


def scot_multi_aoi(grnd_df, prop_df, threshold=0.25, base_reward=100., beta=2.,
                   stats=True, verbose=False):
    """
    SpaceNet Change and Object Tracking (SCOT) metric,
    for a SpaceNet 7 submission with multiple AOIs.
    Input dataframes should have "aoi", "timestep", "id", & "geometry" columns.
    """

    # Get list of AOIs from ground truth dataframe
    aois = sorted(list(grnd_df.aoi.drop_duplicates()))

    # Evaluate SCOT metric for each AOI
    cumulative_score = 0.
    all_stats = {}
    for i, aoi in enumerate(aois):
        if verbose:
            print()
            print('%i / %i: AOI %s' % (i + 1, len(aois), aoi))
        grnd_df_one_aoi = grnd_df.loc[grnd_df.aoi == aoi].copy()
        prop_df_one_aoi = prop_df.loc[prop_df.aoi == aoi].copy()
        score_one_aoi, stats_one_aoi = scot_one_aoi(
            grnd_df_one_aoi, prop_df_one_aoi,
            threshold=threshold,
            base_reward=base_reward,
            beta=beta, stats=True, verbose=verbose)
        cumulative_score += score_one_aoi
        all_stats[aoi] = stats_one_aoi

    # Return combined SCOT metric score
    score = cumulative_score / len(aois)
    if verbose:
        print('Overall score: %f' % score)
    if stats:
        return score, all_stats
    else:
        return score
