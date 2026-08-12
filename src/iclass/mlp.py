"""Routines related to the RF classifier for IRF classes of the lst-irf-classes
module.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance


logger = logging.getLogger(__name__)


def feature_importance_mlp(
    feature_names: list,
    mlp_model: dict,
    x,
    y,
    log_reco_energy,
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

    models = mlp_model["models"]
    energy_edges = mlp_model["energy_edges"]

    energy_ids = np.digitize(log_reco_energy, energy_edges)

    feature_importances = {}

    for energy_id, clf in models.items():

        selection = energy_ids == energy_id

        if selection.sum() == 0:
            continue

        result = permutation_importance(
            clf,
            x.loc[selection],
            y.loc[selection],
            n_repeats=10,
            random_state=0,
            n_jobs=-1,
        )

        df = pd.DataFrame({
            "Feature": clf.feature_names_in_,
            "Importance": result.importances_mean,
        }).sort_values(
            by="Importance",
            ascending=False,
        )

        feature_importances[energy_id] = df

    return feature_importances




def train_mlp(
    df_train: pd.DataFrame,
    config: dict = None,
    ebinsdec: int = 3,
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

    energy_edges = np.arange(
        df_train["log_reco_energy"].min(),
        df_train["log_reco_energy"].max() + 1e-6,
        step=1 / ebinsdec,
    )

    # Assign each training event to an energy bin
    energy_ids = np.digitize(
        df_train["log_reco_energy"],
        energy_edges,
    ) - 1

    models = {}

    for energy_id in np.unique(energy_ids):

        selection = energy_ids == energy_id

        if not np.any(selection):
            continue

        min_energy = energy_edges[energy_id]
        max_energy = energy_edges[energy_id + 1]

        if config:
            regressor_args = config['mlp_regressor_args']
            features = config['mlp_regressor_features']
            clf = model(**regressor_args)

            logger.info("Using features: %s", repr(features))
            logger.info("Training MLP Regressor for reco_offset ...")

        else:
            features = df_train.columns.drop('reco_offset')
            clf = model()
            logger.info("No config provided, using all columns as features.")
            logger.info("Training MLP Regressor with default settings ...")


        logger.info(
            "Training MLP for energy bin %d "
            "(%.3f <= log10(E) < %.3f) with %d events",
            energy_id,
            min_energy,
            max_energy,
            selection.sum(),
        )
        clf.fit(
            df_train.loc[selection, features],
            df_train.loc[selection, "reco_offset"],
        )

        # Store model using the energy-bin ID
        models[energy_id] = clf


    logger.info("Trained %d MLP models.", len(models))
    return {"models": models, "energy_edges": energy_edges}


def apply_mlp(sample: pd.DataFrame, mlp_model: dict, energy_edges: np.ndarray) -> pd.DataFrame:
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
    sample.loc[:, "pred_reco_offset"] = np.nan

    # Use THE SAME energy edges that were used during training
    energy_ids = np.digitize(
        sample["log_reco_energy"],
        energy_edges,
    )

    for energy_id in np.unique(energy_ids):

        selection = energy_ids == energy_id

        if not np.any(selection):
            continue

        # No model available for this energy bin
        if energy_id not in mlp_model:
            continue

        mlp = mlp_model[energy_id]

        features = mlp.feature_names_in_

        sample.loc[selection, "pred_reco_offset"] = mlp.predict(
            sample.loc[selection, features]
        )

    return sample

