"""Routines related to the RF classifier for IRF classes of the lst-irf-classes
module.
"""

import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def feature_importance(
    feature_names: list,
    clf: RandomForestClassifier
) -> pd.DataFrame:
    """Function to estimate the importance of features in a Random Forest
    classifier based on the Gini index.

    Parameters
    ----------
    feature_names : list
        Name of the columns of the dataframe used to train the RF.
    clf : RandomForestClassifier
        Trained RF for which the importance of features shall be checked.

    Returns
    -------
    pd.DataFrame
        Ranked importance of the features.
    """

    importances = clf.feature_importances_
    feature_importances = pd.DataFrame({'Feature': feature_names,
                                        'Importance': importances})
    feature_importances = feature_importances.sort_values(
                                                          by='Importance',
                                                          ascending=False
                                                          )

    return feature_importances


def train_rf(
    df_train: pd.DataFrame,
    config: dict = None
) -> RandomForestClassifier:
    """
    Train a Random Forest Regressor for the classification of irf classes.

    Parameters
    ----------
    train: `pandas.DataFrame`
        Data frame of events to train the RF with.
    config: dictionary
        config file containing the features for the RF training.

    Returns
    -------
    The trained classifier object.
    """

    model = RandomForestClassifier
    logger.info("Number of events for training: %d", df_train.shape[0])

    if config:
        classifier_args = config['random_forest_args']
        features = config['random_forest_features']
        clf = model(**classifier_args)

        logger.info("Given features: %s", repr(features))
        logger.info("Training Random Forest Classifier for PSF Classes ...")

        clf.fit(df_train[features],
                df_train['psf_class'])

    else:
        clf = model()

        logger.info("No features provided, use all columns.")
        logger.info("No RF settings provided, use scikit-learn defaults.")
        logger.info("Training Random Forest Calssifier for PSF Classes...")

        clf.fit(df_train.drop(columns=['psf_class']),
                df_train['psf_class'])

    logger.info("Model %s trained!", type(model).__name__)
    return clf


def apply_rf(sample: pd.DataFrame, rf: RandomForestClassifier) -> pd.DataFrame:
    """
    Apply the pre-trained random forest to the given data frame

    Parameters
    ----------
    sample: pd.DataFrame
        Data frame to apply the random forest to.
    rf: RandomForestClassifier
        Pre-trained random forest

    Returns
    -------
    pd.DataFrame:
        Original data frame with the added 'reco_psf_class' column
        containing the random forest predictions
    """
    features = rf.feature_names_in_
    sample.loc[:, 'reco_psf_class'] = rf.predict(sample[features])

    return sample
