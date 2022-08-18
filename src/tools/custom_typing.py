import numpy as np
import pandas as pd
from typing import Tuple, Union, Generator, Optional
from typing_extensions import Protocol

Number = Union[int, float]
Dataset = Union[pd.DataFrame, pd.Series, np.ndarray]


# Cross validation typing:

class SklearnCrossValidation(Protocol):
    def split(
            self, x, y=None, **kwargs) -> Tuple[Dataset, Dataset]:
        pass


CVGenerator = Generator[
    Tuple[int, Dataset, Dataset, Optional[Dataset], Optional[Dataset]],
    None,
    None]
