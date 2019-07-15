from tensorflow.keras import backend as K
from tensorflow import keras


def get_metrics(framework, config):
    """Load model training metrics from a config file for a specific framework.
    """
    training_metrics = []
    validation_metrics = []

    # TODO: enable passing kwargs to these metrics. This will require
    # writing a wrapper function that'll receive the inputs from the model
    # and pass them along with the kwarg to the metric function.
    if config['training']['metrics'].get('training', []) is None:
        training_metrics = []
    else:
        for m in config['training']['metrics'].get('training', []):
            training_metrics.append(metric_dict[m])
    if config['training']['metrics'].get('validation', []) is None:
        validation_metrics = []
    else:
        for m in config['training']['metrics'].get('validation', []):
            validation_metrics.append(metric_dict[m])

    return {'train': training_metrics, 'val': validation_metrics}


def dice_coef_binary(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 2 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'),
                                   num_classes=2)[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))


def precision(y_true, y_pred):
    """Precision for foreground pixels.

    Calculates pixelwise precision TP/(TP + FP).

    """
    # count true positives
    truth = K.round(K.clip(y_true, K.epsilon(), 1))
    pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1))
    true_pos = K.sum(K.cast(K.all(K.stack([truth, pred_pos], axis=2), axis=2),
                            dtype='float64'))
    pred_pos_ct = K.sum(pred_pos) + K.epsilon()
    precision = true_pos/pred_pos_ct

    return precision


def recall(y_true, y_pred):
    """Precision for foreground pixels.

    Calculates pixelwise recall TP/(TP + FN).

    """
    # count true positives
    truth = K.round(K.clip(y_true, K.epsilon(), 1))
    pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1))
    true_pos = K.sum(K.cast(K.all(K.stack([truth, pred_pos], axis=2), axis=2),
                            dtype='float64'))
    truth_ct = K.sum(K.round(K.clip(y_true, K.epsilon(), 1)))
    if truth_ct == 0:
        return 0
    recall = true_pos/truth_ct

    return recall


def f1_score(y_true, y_pred):
    """F1 score for foreground pixels ONLY.

    Calculates pixelwise F1 score for the foreground pixels (mask value == 1).
    Returns NaN if the model does not identify any foreground pixels in the
    image.

    """

    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    # Calculate f1_score
    f1_score = 2 * (prec * rec) / (prec + rec)

    return f1_score


# the keras metrics functions _should_ also work if provided with a
# (y_true, y_pred) pair from pytorch, so I'll use those for both.
metric_dict = {
    'accuracy': keras.metrics.binary_accuracy,
    'binary_accuracy': keras.metrics.binary_accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'categorical_accuracy': keras.metrics.categorical_accuracy,
    'cosine': keras.metrics.cosine_proximity,
    'cosine_proximity': keras.metrics.cosine_proximity,
    'hinge': keras.metrics.hinge,
    'squared_hinge': keras.metrics.squared_hinge,
    'kld': keras.metrics.kullback_leibler_divergence,
    'kullback_leibler_divergence': keras.metrics.kullback_leibler_divergence,
    'mae': keras.metrics.mean_absolute_error,
    'mean_absolute_error': keras.metrics.mean_absolute_error,
    'mse': keras.metrics.mean_squared_error,
    'mean_squared_error': keras.metrics.mean_squared_error,
    'msle': keras.metrics.mean_squared_logarithmic_error,
    'mean_squared_logarithmic_error': keras.metrics.mean_squared_logarithmic_error,
    'sparse_categorical_accuracy': keras.metrics.sparse_categorical_accuracy,
    'top_k_categorical_accuracy': keras.metrics.top_k_categorical_accuracy
}
