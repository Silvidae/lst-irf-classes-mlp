"""Tests for the routines of the RF module (rf_func).
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from iclass.rf import feature_importance, train_rf, apply_rf


class TestFeatureImportance(unittest.TestCase):
    """Class for testing the RF feaature importance function.
    """

    def test_feature_importance(self):
        """Testing the feature importance function.
        """
        # Define mock feature names and importances
        feature_names = ['feature1', 'feature2', 'feature3']
        importances = [0.3, 0.5, 0.2]

        # Create a mock classifier with predefined feature importances
        mock_clf = MagicMock()
        mock_clf.feature_importances_ = importances

        # Call the function with the mock classifier
        result = feature_importance(feature_names, mock_clf)

        # Define expected output
        expected_df = pd.DataFrame({
            'Feature': ['feature2', 'feature1', 'feature3'],
            'Importance': [0.5, 0.3, 0.2]
        }).reset_index(drop=True)

        # Check if the result matches the expected DataFrame
        pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                      expected_df)


class TestTrainRF(unittest.TestCase):
    """Class for testing the RF training function.
    """

    @patch('iclass.rf.RandomForestClassifier')
    def test_train_rf_with_config(self, mock_rf):
        """Testing the rf training with configuration.
        """
        # Mock configuration and training data
        config = {
            'random_forest_args': {'n_estimators': 10, 'max_depth': 5},
            'random_forest_features': ['feature1', 'feature2']
        }
        df_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'psf_class': [0, 1, 0]
        })

        # Run the train_rf function with config
        df_train_original = df_train.copy(deep=True)
        train_rf(df_train, config)

        # Ensure df_train was not modified
        pd.testing.assert_frame_equal(df_train, df_train_original,
                                      check_dtype=False, check_like=True)

        # Assert RandomForestClassifier was initialized with the given
        # arguments
        mock_rf.assert_called_once_with(**config['random_forest_args'])

        # Get mock instance of the RandomForestClassifier
        mock_clf = mock_rf()

        # Verify clf.fit was called with the specified features and target
        actual_args = mock_clf.fit.call_args[0]
        expected_features = df_train[['feature1', 'feature2']]
        expected_target = df_train['psf_class']

        pd.testing.assert_frame_equal(actual_args[0],
                                      expected_features,
                                      check_dtype=False)
        pd.testing.assert_series_equal(actual_args[1],
                                       expected_target,
                                       check_dtype=False)

    @patch('iclass.rf.RandomForestClassifier')
    def test_train_rf_without_config(self, mock_rf):
        """Testing the rf training without configuration.
        """
        # Mock training data with additional features
        df_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'psf_class': [0, 1, 0]
        })

        # Run the train_rf function without a config
        df_train_original = df_train.copy(deep=True)
        train_rf(df_train, config=None)

        # Ensure df_train was not modified
        pd.testing.assert_frame_equal(df_train, df_train_original,
                                      check_dtype=False, check_like=True)

        # Assert RandomForestClassifier was initialized with default arguments
        mock_rf.assert_called_once_with()

        # Get mock instance of the RandomForestClassifier
        mock_clf = mock_rf()

        # Verify clf.fit was called with the specified features and target
        actual_args = mock_clf.fit.call_args[0]
        expected_features = df_train[['feature1', 'feature2']]
        expected_target = df_train['psf_class']

        pd.testing.assert_frame_equal(actual_args[0],
                                      expected_features,
                                      check_dtype=False)
        pd.testing.assert_series_equal(actual_args[1],
                                       expected_target,
                                       check_dtype=False)

    def test_apply_rf(self):
        """Function to test whether the RF is correctly applied.
        """
        rf = Mock()
        rf.feature_names_in_ = ['x', 'y']
        rf.predict = Mock(return_value=[1, 2, 3])

        data = {
            "x": [0, 0, 0],
            "y": [1, 1, 1],
        }
        sample = pd.DataFrame(data=data)
        result = apply_rf(sample, rf)

        self.assertListEqual(
            result['reco_psf_class'].to_list(),
            rf.predict.return_value
        )
