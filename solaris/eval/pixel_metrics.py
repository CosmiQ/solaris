import time
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


###############################################################################
def compute_iou(truth_mask, prop_mask, verbose=False):
    """
    Compute pixel-wise intersection over union

    Notes
    -----
    Multiply truth_mask by 2, and subtract.  Make sure arrays are clipped
    so that overlapping regions don't cause problems

    Arguments
    ---------
    truth_mask : np array
        2-D array of ground truth.
    prop_mask : np array
        2-D array of proposals.
    verbose : bool
        Switch to print relevant values

    Returns
    -------
    iou : float
        Intersection over union of ground truth and proposal
    """

    truth_mask_clip = np.clip(truth_mask, 0, 1).astype(float)
    prop_mask_clip = np.clip(prop_mask, 0, 1).astype(float)
    # subtract array
    sub_mask = 2*prop_mask_clip - truth_mask_clip
    add_mask = prop_mask_clip + truth_mask_clip

    # true pos = 1, false_pos = 2, true_neg = 0, false_neg = -1
    true_pos_count = len(np.where(sub_mask == 1)[0])
    union = len(np.where(add_mask > 0)[0])
    intersection = true_pos_count

    iou = 1. * intersection / union

    if verbose:
        print("intersection:", intersection)
        print("union:", union)
        print("iou:", iou)

    return iou


###############################################################################
def compute_f1(truth_mask, prop_mask, show_plot=False, im_file='',
               show_colorbar=False, plot_file='', dpi=200, verbose=False):
    """
    Compute pixel-wise f1 score

    Notes
    -----
    Find true pos, false pos, true neg, false neg as well as f1 score.
    Multiply truth_mask by 2, and subtract.  Make sure arrays are clipped
    so that overlapping regions don't cause problems.

    Arguments
    ---------
    truth_mask : np array
        2-D array of ground truth.
    prop_mask : np array
        2-D array of proposals.
    show_plot : bool
        Switch to plot the outputs. Defaults to ``False``.
    im_file : str
        Image file corresponding to the masks. Ignored if show_plot == False.
        Defaults to ``''``.
    show_colorbar : bool
        Switch to show colorbar. Ignored if show_plot == False.
        Defaults to ``False``.
    plot_file : str
        Output file if plotting. Ignored if show_plot == False.
        Defaults to ``''``.
    dpi : int
        Dots per inch for plotting. Ignored if show_plot == False.
        Defaults to ``200``.
    verbose : bool
        Switch to print relevant values

    Returns
    -------
    output : tuple
        Tuple containing [f1, precision, recall]
    """

    truth_mask_clip = np.clip(truth_mask, 0, 1).astype(float)
    prop_mask_clip = np.clip(prop_mask, 0, 1).astype(float)
    # subtract array
    sub_mask = 2*prop_mask_clip - truth_mask_clip
    # sub_mask2 = prop_mask_clip - truth_mask_clip

    # true pos = 1, false_pos = 2, true_neg = 0, false_neg = -1
    n_pos = len(np.where(truth_mask_clip == 1)[0])
    true_pos_count = len(np.where(sub_mask == 1)[0])
    false_pos_count = len(np.where(sub_mask == 2)[0])
    true_neg_count = len(np.where(sub_mask == 0)[0])
    false_neg_count = len(np.where(sub_mask == -1)[0])

    if (n_pos > 0) and (true_pos_count > 0):
        precision = float(true_pos_count) / float( true_pos_count + false_pos_count)
        recall = float(true_pos_count) / float( true_pos_count + false_neg_count)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision, recall, f1 = 0, 0, 0

    if verbose:
        print("mask.shape:\t", truth_mask.shape)
        print("num pixels:\t", truth_mask.size)
        print("false_neg:\t", false_neg_count)
        print("false_pos:\t", false_pos_count)
        print("true_neg:\t", true_neg_count)
        print("true_pos:\t", true_pos_count)
        print("precision:\t", precision)
        print("recall:\t\t", recall)
        print("F1 score:\t", f1)

    if show_plot:

        fontsize = 6
        t0 = time.time()
        title = "Precision: " + str(np.round(precision, 3)) \
                + "  Recall: " + str(np.round(recall, 3)) \
                + "  F1: " + str(np.round(f1, 3))

        if show_colorbar:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,
                                                         sharey=True,
                                                         figsize=(6, 6))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True,
                                                sharey=True,
                                                figsize=(9.5, 3))
        # fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=True,
        #    sharey=True, figsize=(13,4))
        plt.suptitle(title, fontsize=fontsize)

        # ground truth
        if len(im_file) > 0:
            # raw image
            ax1.imshow(cv2.imread(im_file, 1))
            # ground truth
            # set zeros to nan
            palette = plt.cm.gray
            palette.set_over('orange', 1.0)
            palette.set_over('orange', 1.0)

            z = truth_mask.astype(float)
            z[z == 0] = np.nan
            ax1.imshow(z, cmap=palette, alpha=0.5,
                       norm=matplotlib.colors.Normalize(
                               vmin=0.5, vmax=0.9, clip=False))
            ax1.set_title('truth_mask_clip + image', fontsize=fontsize)
        else:
            ax1.imshow(truth_mask_clip)
            ax1.set_title('truth_mask_clip', fontsize=fontsize)
        ax1.axis('off')

        # proposal mask
        ax2.imshow(prop_mask_clip)
        ax2.axis('off')
        ax2.set_title('prop_mask_clip', fontsize=fontsize)

        # mask
        if show_colorbar:
            z = ax3.pcolor(sub_mask)
            fig.colorbar(z)
            ax4.axis('off')

        else:
            ax3.imshow(sub_mask)
        # z = ax3.pcolor(sub_mask2)
        ax3.axis('off')
        ax3.set_title('subtract_mask', fontsize=fontsize)

        # plt.tight_layout()
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(top=0.8)

        if len(plot_file) > 0:
            plt.savefig(plot_file, dpi=dpi)
        print("Time to create and save F1 plots:", time.time() - t0, "seconds")

        plt.show()

    output = (f1, precision, recall)
    return output


###############################################################################
def _get_neighborhood_limits(row, col, h, w, rho=3):
    '''Get neighbors of point p with pixel coords row, col'''

    rowmin = max(0, row-rho)
    rowmax = min(h, row + rho)
    colmin = max(0, col-rho)
    colmax = min(w, col + rho)

    return rowmin, rowmax, colmin, colmax


###############################################################################
def compute_relaxed_f1(truth_mask, prop_mask, radius=3, verbose=False):
    """
    Compute relaxed f1 score

    Notes
    -----
    Also find relaxed precision, recall, f1.
    http://www.cs.toronto.edu/~fritz/absps/road_detection.pdf
      "completenetess represents the fraction of true road pixels that are
      within ρ pixels of a predicted road pixel, while correctness measures
      the fraction of predicted road pixels that are within ρ pixels of a
      true road pixel."
    https://arxiv.org/pdf/1711.10684.pdf
     The relaxed precision is defined as the fraction of number of pixels
     predicted as road
     within a range of ρ pixels from pixels labeled as road. The
     relaxed recall is the fraction of number of pixels labeled as
     road that are within a range of ρ pixels from pixels predicted
     as road.
    http://ceur-ws.org/Vol-156/paper5.pdf

    Arguments
    ---------
    truth_mask : np array
        2-D array of ground truth.
    prop_mask : np array
        2-D array of proposals.
    radius : int
        Radius in pixels to use for relaxed f1.
    verbose : bool
        Switch to print relevant values

    Returns
    -------
    output : tuple
        Tuple containing [relaxed_f1, relaxed_precision, relaxed_recall]

    Example usage
    -------------
    h,w = 50, 50
    truth_mask = np.random.randint(2, size=(h,w))
    prop_mask = np.random.randint(2, size=(h,w))
    compute_relaxed_f1(truth_mask, prop_mask, radius=3, verbose=True)
    """

    truth_mask_clip = np.clip(truth_mask, 0, 1).astype(float)
    prop_mask_clip = np.clip(prop_mask, 0, 1).astype(float)

    # true pos = 1, false_pos = 2, true_neg = 0, false_neg = -1
    n_truth = len(np.where(truth_mask_clip == 1)[0])
    n_prop = len(np.where(prop_mask_clip == 1)[0])

    # iterate through truth pixels
    precision_count = 0
    recall_count = 0
    h, w = truth_mask.shape
    for row in range(h):
        for col in range(w):
            truth_val = truth_mask_clip[row][col]
            prop_val = prop_mask_clip[row][col]
            # get window limits
            rowmin, rowmax, colmin, colmax = _get_neighborhood_limits(
                row, col, h, w, rho=radius)
            # get windows
            truth_win = truth_mask_clip[rowmin:rowmax, colmin:colmax]
            prop_win = prop_mask_clip[rowmin:rowmax, colmin:colmax]

            # add precision_count if proposal is within the radius of a gt node
            if prop_val == 1:
                if np.max(truth_win) > 0:
                    precision_count += 1

            if truth_val == 1:
                if np.max(prop_win) > 0:
                    recall_count += 1

    # get fractions
    if n_truth == 0:
        relaxed_recall = 0
    else:
        relaxed_recall = 1. * recall_count / n_truth
    if n_prop == 0:
        relaxed_precision = 0
    else:
        relaxed_precision = 1. * precision_count / n_prop

    if (relaxed_recall > 0) and (relaxed_precision > 0):
        f1 = 2 * relaxed_precision * relaxed_recall \
            / (relaxed_precision + relaxed_recall)
    else:
        f1 = 0

    if verbose:
        print("mask.shape:\t", truth_mask.shape)
        print("num pixels:\t", truth_mask.size)
        print("precision:\t", relaxed_precision)
        print("recall:\t\t", relaxed_recall)
        print("rF1 score:\t", f1)

    output = (f1, relaxed_precision, relaxed_recall)
    return output
