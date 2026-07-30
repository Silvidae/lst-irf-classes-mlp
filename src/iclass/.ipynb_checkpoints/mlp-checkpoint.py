"""Routines related to the RF classifier for IRF classes of the lst-irf-classes
module.
"""

import logging
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance


logger = logging.getLogger(__name__)


def feature_importance_mlp(
    feature_names: list,
    clf: MLPRegressor,
    x,
    y
) -> pd.DataFrame:
    """Estimate feature importance for an MLP regressor using permutation importance.

    Parameters
    ----------
    feature_names : list
        Names of the columns of the dataframe used to train the model.
    clf : MLPRegressor
        Trained MLP regressor.
    X : array-like or DataFrame
        Feature dataset used to evaluate importance.
    y : array-like
        Target values.

    Returns
    -------
    pd.DataFrame
        Ranked importance of the features.
    """

    result = permutation_importance(
        clf,
        x,
        y,
        n_repeats=10,
        random_state=0,
        n_jobs=-1
    )

    importances = result.importances_mean

    feature_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_importances = feature_importances.sort_values(
        by="Importance",
        ascending=False
    )

    return feature_importances




def train_mlp(
    df_train: pd.DataFrame,
    config: dict = None
) -> MLPRegressor:
    """
    Train the MLP Regressor for the definition of irf types.

    Parameters
    ----------
    train: `pandas.DataFrame`
        Data frame of events to train the MLP with.
    config: dictionary
        config file containing the features for the MLP training.

    Returns
    -------
    The trained classifier object.
    """

    model = MLPRegressor
    logger.info("Number of events for training: %d", df_train.shape[0])

    if config:
        regressor_args = config.get('mlp_regressor_args', {})
        features = config.get('mlp_regressor_features', df_train.columns.drop('reco_offset').tolist())
        clf = model(**regressor_args)

        logger.info("Using features: %s", repr(features))
        logger.info("Training MLP Regressor for reco_offset ...")

        clf.fit(df_train[features], df_train['reco_offset'])

    else:
        features = df_train.columns.drop('reco_offset')
        clf = model()
        logger.info("No config provided, using all columns as features.")
        logger.info("Training MLP Regressor with default settings ...")
        
        clf.fit(df_train[features],
                df_train['reco_offset'])

    logger.info("Model %s trained!", type(clf).__name__)
    return clf


def apply_mlp(sample: pd.DataFrame, mlp: MLPRegressor) -> pd.DataFrame:
    """
    Apply the pre-trained regressor to the given data frame

    Parameters
    ----------
    sample: pd.DataFrame
        Data frame to apply the regressor to.
    mlp: MLPRegressor
        Pre-trained MLP regressor

    Returns
    -------
    pd.DataFrame:
        Original data frame with the added 'reco_psf_class' column
        containing the random forest predictions
    """
    features = mlp.feature_names_in_
    # "reco_psf_class" for the RF classificator
    sample.loc[:, 'pred_reco_offset'] = mlp.predict(sample[features]) 

    return sample
