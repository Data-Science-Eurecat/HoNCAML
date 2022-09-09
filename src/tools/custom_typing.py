import numpy.typing as npt
import pandas as pd
from typing import Tuple, Union, Generator, Optional, List
from typing_extensions import Protocol

# Generic typing

Number = Union[int, float]
Array = npt.ArrayLike
Dataset = Union[pd.DataFrame, pd.Series, Array]
StrList = List[str]


# Cross validation typing

class SklearnCrossValidation(Protocol):
    def split(
            self, x, y=None, **kwargs) -> Tuple[Dataset, Dataset]:
        pass


CVGenerator = Generator[
    Tuple[int, Dataset, Dataset, Optional[Dataset], Optional[Dataset]],
    None,
    None]
