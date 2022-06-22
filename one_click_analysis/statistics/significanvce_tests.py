import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

from one_click_analysis.feature_processing.attributes.attribute import AttributeDataType


def compute_p_value(
    x: np.ndarray,
    y: np.ndarray,
    data_type_x: AttributeDataType,
    data_type_y: AttributeDataType,
):
    """Convenience function to compute the p values."""
    if (
        data_type_x == AttributeDataType.NUMERICAL
        and data_type_y == AttributeDataType.NUMERICAL
    ):
        return pearson_test(x, y)
    elif (
        data_type_x == AttributeDataType.CATEGORICAL
        and data_type_y == AttributeDataType.CATEGORICAL
    ):
        return chisquare_test(x, y)
    elif (
        data_type_x == AttributeDataType.CATEGORICAL
        and data_type_y == AttributeDataType.NUMERICAL
    ):
        return ttest(x, y)
    elif (
        data_type_x == AttributeDataType.NUMERICAL
        and data_type_y == AttributeDataType.CATEGORICAL
    ):
        return ttest(y, x)
    else:
        ValueError("Only combinations of numerical and categorical datatypes supported")


def chisquare_test(x: np.ndarray, y: np.ndarray):
    """Compute chi square p value for binary-binary case."""
    indices_x_0 = np.where(x == 0)[0]
    indices_x_1 = np.where(x == 1)[0]

    num_0_0 = np.unique(y[indices_x_0], return_counts=True)[1][0]
    num_0_1 = np.unique(y[indices_x_0], return_counts=True)[1][1]
    num_1_0 = np.unique(y[indices_x_1], return_counts=True)[1][0]
    num_1_1 = np.unique(y[indices_x_1], return_counts=True)[1][1]

    p_val = chi2_contingency([[num_0_0, num_0_1], [num_1_0, num_1_1]])[1]
    return p_val


def pearson_test(x: np.ndarray, y: np.ndarray):
    """Compute pearson p value for continuous-continuous data"""
    return pearsonr(x, y)[1]


def ttest(x_binary: np.array, y_cont: np.array):
    """Compute ttest p value for binary-continuous or continuous-binary data

    :param x_binary: binary data
    :param y_cont: continuous data
    """
    indices_x_0 = np.where(x_binary == 0)[0]
    indices_x_1 = np.where(x_binary == 1)[0]

    y_vals_x0 = y_cont[indices_x_0]
    y_vals_x1 = y_cont[indices_x_1]

    return ttest_ind(y_vals_x0, y_vals_x1)[1]
