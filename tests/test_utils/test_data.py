import os
import pandas as pd
from solaris.data import data_dir
from solaris.utils.data import make_dataset_csv


class TestMakeDatasetCSV(object):
    """Test sol.utils.data.make_dataset_csv()."""

    def test_with_regex(self):
        output_df = make_dataset_csv(
            im_dir=os.path.join(data_dir, 'rastertile_test_expected'),
            label_dir=os.path.join(data_dir, 'vectortile_test_expected'),
            match_re=r'([0-9]{6}_[0-9]{7})',
            output_path=os.path.join(data_dir, 'tmp.csv'))
        assert len(output_df) == 100
        im_substrs = output_df['image'].str.extract(r'([0-9]{6}_[0-9]{7})')
        label_substrs = output_df['label'].str.extract(r'([0-9]{6}_[0-9]{7})')
        assert im_substrs.equals(label_substrs)
        os.remove(os.path.join(data_dir, 'tmp.csv'))

    def test_no_regex_get_error(self):
        try:
            # this *should* throw a ValueError
            _ = make_dataset_csv(
                im_dir=os.path.join(data_dir, 'rastertile_test_expected'),
                label_dir=os.path.join(data_dir, 'vectortile_test_expected'))
            assert False  # it should never get here
        except ValueError:
            assert True

    def test_no_regex_skip_mismatch(self):
        # this should generate an empty df because it doesn't use a regex to
        # match images, it uses the full filename, which is different between
        # the two sets.
        output_df = make_dataset_csv(
            im_dir=os.path.join(data_dir, 'rastertile_test_expected'),
            label_dir=os.path.join(data_dir, 'vectortile_test_expected'),
            ignore_mismatch='skip',
            output_path=os.path.join(data_dir, 'tmp.csv'))

        assert len(output_df) == 0
        os.remove(os.path.join(data_dir, 'tmp.csv'))

    def test_catch_no_labels(self):
        # make sure it generates an error if you call the function but don't
        # give it labels for a training set
        try:
            _ = make_dataset_csv(
                im_dir=os.path.join(data_dir, 'rastertile_test_expected'),
                ignore_mismatch='skip', stage='train',
                output_path=os.path.join(data_dir, 'tmp.csv'))
            assert False
        except ValueError:
            assert True

    def test_infer_dataset(self):

        output_df = make_dataset_csv(
            im_dir=os.path.join(data_dir, 'rastertile_test_expected'),
            ignore_mismatch='skip', stage='infer',
            output_path=os.path.join(data_dir, 'tmp.csv'))

        assert len(output_df) == 100
        assert len(output_df.columns) == 1
        os.remove(os.path.join(data_dir, 'tmp.csv'))
