import numpy as np


def remove_outliers_IQR(
    x_values: np.ndarray, y_values: np.ndarray, apply_on_binary: bool = False
):
    """Remove the rows with outleirs in values_x and values_y.
    When there is an outlier in a row in values_x, the same row will be removed in
    values_y. The same applies if there is an outlier in a row in values_y.

    :param x_values: x-values
    :param y_values: y-values
    :param apply_on_binary: If True, also apply on binary values.
    """

    # check if binary
    x_binary = False
    y_binary = False
    if apply_on_binary:
        unique_x = np.unique(x_values)
        unique_y = np.unique(y_values)
        num_els_x = len(unique_x)
        if np.isnan(unique_x).any():
            num_els_x = num_els_x - 1
        num_els_y = len(unique_y)
        if np.isnan(unique_y).any():
            num_els_y = num_els_y - 1

        if num_els_x < 3:
            x_binary = True
        if num_els_y < 3:
            y_binary = True

    if x_binary and y_binary:
        return x_values, y_values

    # If there are only two values
    bool_inliers_x = bool_inliers_y = np.full(len(x_values), True, dtype=bool)

    if not x_binary:
        q1_x = np.quantile(x_values, 0.25)
        q3_x = np.quantile(x_values, 0.75)
        iqr_x = q3_x - q1_x
        outlier_idxs_x = np.where(
            (x_values < (q1_x - 1.5 * iqr_x)) & (x_values < (q3_x + 1.5 * iqr_x))
        )[0]
        bool_inliers_x[outlier_idxs_x] = False

    if not y_binary:
        q1_y = np.quantile(y_values, 0.25)
        q3_y = np.quantile(y_values, 0.75)
        iqr_y = q3_y - q1_y
        outlier_idxs_y = np.where(
            (y_values < (q1_y - 1.5 * iqr_y)) & (y_values > (q3_y + 1.5 * iqr_y))
        )[0]
        bool_inliers_y[outlier_idxs_y] = False

    bool_inliers = bool_inliers_x & bool_inliers_y

    return x_values[bool_inliers], y_values[bool_inliers]
