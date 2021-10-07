"""
Base algorithm to find outliers.
"""
from sklearn.base import TransformerMixin


class FittedAlgorithmError(Exception):
    """Exception raised when outlier algorithm is not fitted."""

    def __init__(self, message):
        self.message = message


class Base(TransformerMixin):
    """Base class to implement outlier algorithms.

    Any algorithm used to find outlier must extend this class and implement 
    'fit' method.

    Parameters
    ----------
    mode : str, {'rolling', 'expanding'}
        Defines whether to apply the filter in rolling mode or in expanding
        mode.
    impute : str
        Value to set the filtered values.
    window : int
        Number of windows for the rolling/expanding modes.

    Attributes
    ----------
    original_ts : array-like
        Original time series.
    filtered_ts_ : array-like
        Filtered time series.
    outlier_mask_ : array-like
        Array-like containing True positives (outliers) and True negatives 
        (normal values).
    __fitted__ : bool
        True if the algorithm has been fitted.
    """
    def __init__(self, mode, impute, window) -> None:
        self.mode = mode
        self.impute = impute
        self.window = window
        
        # attributes
        self.original_ts = None
        self.filtered_ts_ = None
        self.outlier_mask_ = None
        self.__fitted__ = False

    def fit(self, X):
        raise NotImplementedError()

    def transform(self, X):
        """
        Return the filtered time series as well as the mask where outliers can 
        be found.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        filtered_ts_ : array-like
            Filtered time series.
        outlier_mask_ : array-like
            Array-like indicating the values marked as outlier (1).
        """
        if not self.__fitted__:
            FittedAlgorithmError(f'Please, fit the algotihm first.')
        
        return self.filtered_ts_, self.outlier_mask_

    def plot_report(self):
        """
        Plot a report with the 
        """
        if not self.__fitted__:
            FittedAlgorithmError(f'Please, fit the algotihm first.')
        
        pass

    def check_is_fitted(self):
        return self.__fitted__