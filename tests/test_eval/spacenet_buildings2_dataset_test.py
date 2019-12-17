import os
from solaris.eval.challenges import spacenet_buildings_2
import solaris
import subprocess
import pandas as pd


class TestEvalSpaceNetBuildings2(object):
    """Tests for the ``spacenet_buildings_2`` function."""

    def test_scoring(self):
        """Test a round of scoring."""
        # load predictions
        pred_results = pd.read_csv(os.path.join(solaris.data.data_dir,
                                                'SN2_test_results.csv'))
        pred_results_full = pd.read_csv(
            os.path.join(solaris.data.data_dir,
                         'SN2_test_results_full.csv'))
        results_df, results_df_full = spacenet_buildings_2(
            os.path.join(solaris.data.data_dir, 'SN2_sample_preds.csv'),
            os.path.join(solaris.data.data_dir, 'SN2_sample_truth.csv')
            )

        results_df_formatted = results_df.reset_index(drop=True)
        pred_results_full_sorted = (pred_results_full
                                    .sort_values('imageID')
                                    .reset_index(drop=True))
        results_df_full_sorted = (results_df_full
                                  .sort_values('imageID')
                                  .reset_index(drop=True))
        assert almostequal(pred_results, results_df_formatted[pred_results.columns])
        assert almostequal(pred_results_full_sorted, results_df_full_sorted[pred_results_full_sorted.columns])


class TestEvalCLISN2(object):
    """Test the CLI ``spacenet_eval`` function, as applied to SpaceNet2."""

    def test_cli(self):
        """Test a round of scoring using the CLI."""
        pred_results = pd.read_csv(os.path.join(
            solaris.data.data_dir, 'SN2_test_results.csv'))
        pred_results_full = pd.read_csv(os.path.join(
            solaris.data.data_dir, 'SN2_test_results_full.csv'))
        proposal_csv = os.path.join(solaris.data.data_dir,
                                    'SN2_sample_preds.csv')
        truth_csv = os.path.join(solaris.data.data_dir,
                                 'SN2_sample_truth.csv')
        subprocess.call(['spacenet_eval', '--proposal_csv='+proposal_csv,
                         '--truth_csv='+truth_csv,
                         '--challenge=spacenet-buildings2',
                         '--output_file=test_out'])
        test_results = pd.read_csv('test_out.csv')
        full_test_results = pd.read_csv('test_out_full.csv')

        assert pred_results.equals(test_results[pred_results.columns])
        pred_results_full_sorted = pred_results_full.sort_values(by='imageID') \
                                .reset_index(drop=True)
        full_test_results_sorted = full_test_results.sort_values(by='imageID') \
                                .reset_index(drop=True)
        assert pred_results_full_sorted.equals(full_test_results_sorted[pred_results_full_sorted.columns])

        os.remove('test_out.csv')
        os.remove('test_out_full.csv')


def almostequal(df1, df2, epsilon=1E-9):
    """
    Reports whether two dataframes are "almost" equal, allowing for a specified
    rounding error.  Non-identical elements that are not integers or floats
    result in an error.
    """
    if df1.shape != df2.shape:
        return False
    for i in range((df1.shape)[0]):
        for j in range((df1.shape)[1]):
            val1 = df1.iloc[i, j]
            val2 = df2.iloc[i, j]
            if val1 != val2:
                if abs(val2-val1) > 0.5*epsilon*(abs(val1)+abs(val2)):
                    return False
    return True
