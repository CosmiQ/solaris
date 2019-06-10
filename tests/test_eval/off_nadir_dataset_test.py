import os
from solaris.eval.challenges import off_nadir_buildings
import solaris
import subprocess
import pandas as pd


class TestEvalOffNadir(object):
    """Tests for the ``off_nadir`` function."""

    def test_scoring(self):
        """Test a round of scoring."""
        # load predictions
        pred_results = pd.read_csv(os.path.join(solaris.data.data_dir,
                                                'test_results.csv'))
        pred_results_full = pd.read_csv(os.path.join(solaris.data.data_dir,
                                                     'test_results_full.csv'))
        results_df, results_df_full = off_nadir_buildings(
            os.path.join(solaris.data.data_dir, 'sample_preds.csv'),
            os.path.join(solaris.data.data_dir, 'sample_truth.csv')
            )
        assert pred_results.equals(results_df.reset_index())
        assert pred_results_full.equals(results_df_full)


class TestEvalCLI(object):
    """Test the CLI ``spacenet_eval`` function."""

    def test_cli(self):
        """Test a round of scoring using the CLI."""
        pred_results = pd.read_csv(os.path.join(
            solaris.data.data_dir, 'competition_test_results.csv'))
        pred_results_full = pd.read_csv(os.path.join(
            solaris.data.data_dir, 'competition_test_results_full.csv'))
        proposal_csv = os.path.join(solaris.data.data_dir,
                                    'sample_preds_competition.csv')
        truth_csv = os.path.join(solaris.data.data_dir,
                                 'sample_truth_competition.csv')
        subprocess.call(['spacenet_eval', '--proposal_csv='+proposal_csv,
                         '--truth_csv='+truth_csv,
                         '--output_file=test_out'])
        test_results = pd.read_csv('test_out.csv')
        full_test_results = pd.read_csv('test_out_full.csv')

        assert pred_results.equals(test_results)
        assert pred_results_full.sort_values(by='imageID').reset_index(drop=True).equals(
            full_test_results.sort_values(by='imageID').reset_index(drop=True))

        os.remove('test_out.csv')
        os.remove('test_out_full.csv')
