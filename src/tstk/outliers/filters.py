import numpy_ext
import numpy as np
import pandas as pd

from typing import Union
from tstk.outliers.base import Base


class BasicAlgorithm(Base):
    """Basic filtering algorithm.

    Mark as outliers the points that are out of the interval:
        
        (mean - threshold * std, mean + threshold * std ).
    
    Parameters
    ----------
    threshold : int
        Number of standard deviations above which a value will be filtered.
    """
    def __init__(
        self,
        threshold: Union[int, float]=3,
        impute: str='mean',
        mode: str='rolling',
        window: int=261
    ) -> None:
        super(Base, self).__init__(mode=mode, impute=impute, window=window)

        self.threshold = threshold

    def fit(self, X: pd.DataFrame):

        filtered_df = X.copy()

        pd_obj = getattr(filtered_df, self.mode)(self.window)
        mean = pd_obj.mean()
        std = pd_obj.std()

        upper_bound = mean + self.threshold * std
        lower_bound = mean - self.threshold * std
        
        outliers = ~filtered_df.between(lower_bound, upper_bound)
        # fill false positives with 0
        outliers.iloc[:self.window] = np.zeros(shape=self.window)
        
        series = filtered_df.to_frame()
        series['outliers'] = np.array(outliers.astype('int').values)
        series.columns = ['Close', 'Outliers']

        self.__fitted__ = True

        return self
