"""
FNSPID 데이터 로딩 및 시계열 윈도우 생성 모듈.

- CSV 로드 후 시간(날짜) 기준 정렬
- 날짜 기준 train/val/test 분할 (shuffle 없음, 데이터 누수 방지)
- 슬라이딩 윈도우 생성 및 선택 컬럼 정규화
- 기존 core/data_processor.DataLoader 와 인자 호환 가능 (val_ratio=0 이면 동일 사용)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union


def _normalise_selected_columns(
    window_data: np.ndarray,
    columns_to_normalise: List[int],
    single_window: bool = False,
) -> np.ndarray:
    """윈도우 내 첫 시점 값을 기준으로 지정 컬럼만 (value/first - 1) 정규화."""
    normalised_data = []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window = []
        for col_i in range(window.shape[1]):
            if col_i in columns_to_normalise:
                w = window[0, col_i]
                if w == 0:
                    w = 1
                normalised_col = [((float(p) / float(w)) - 1) for p in window[:, col_i]]
            else:
                normalised_col = window[:, col_i].tolist()
            normalised_window.append(normalised_col)
        normalised_window = np.array(normalised_window).T
        normalised_data.append(normalised_window)
    return np.array(normalised_data)


class FNSPIDDataset:
    """
    CSV 로드 → 시간 정렬 → 날짜 기준 분할 → 시계열 윈도우 생성.

    기존 DataLoader 호환:
        data = FNSPIDDataset(path, split=0.85, cols=[...], cols_to_norm=[0,1], pred_len=3)
        x_train, y_train = data.get_train_data(seq_len=50, normalise=True)
        x_test, y_test, y_base = data.get_test_data(seq_len=50, normalise=True, cols_to_norm=[0,1])
    """

    def __init__(
        self,
        filename: str,
        split: float,
        cols: List[str],
        cols_to_norm: List[int],
        pred_len: int,
        val_ratio: float = 0.0,
        date_column: Optional[str] = "Date",
    ):
        """
        Args:
            filename: CSV 파일 경로
            split: 학습 비율 (0~1). val_ratio=0 이면 train=split, test=1-split
            cols: 사용할 컬럼 이름 리스트 (예: ["Close","Volume","Scaled_sentiment"])
            cols_to_norm: 정규화할 컬럼 인덱스 (cols 기준)
            pred_len: 예측 길이 (현재 로직에서는 y가 윈도우 마지막 시점 Close)
            val_ratio: 검증 비율 (0이면 검증 구간 없음, 기존 동작과 동일)
            date_column: 날짜 컬럼명. 있으면 정렬·날짜 기준 분할, None이면 행 순서 그대로 비율 분할
        """
        self.pred_len = pred_len
        self.cols_to_norm = cols_to_norm
        self.date_column = date_column

        dataframe = pd.read_csv(filename)

        # 날짜 컬럼 있으면 파싱 후 오름차순 정렬 (과거 → 미래)
        if date_column and date_column in dataframe.columns:
            dataframe[date_column] = pd.to_datetime(dataframe[date_column])
            dataframe = dataframe.sort_values(date_column).reset_index(drop=True)
            self._date_index = dataframe[date_column].values
        else:
            self._date_index = None
            if date_column:
                import warnings
                warnings.warn(
                    f"date_column='{date_column}' not found. Split by row index (no date-based guarantee)."
                )

        values = dataframe[cols].values.astype(float)
        n = len(values)

        # 분할 인덱스: 시간 순서 유지, shuffle 없음
        if val_ratio <= 0:
            i_train = int(n * split)
            self.data_train = values[:i_train]
            self.data_val = np.array([]).reshape(0, values.shape[1])
            self.data_test = values[i_train:]
        else:
            # train / val / test 순서. test 구간은 기존과 동일하게 맨 뒤 (1-split)
            i_val_start = int(n * (split - val_ratio))
            i_test_start = int(n * split)
            self.data_train = values[:i_val_start]
            self.data_val = values[i_val_start:i_test_start]
            self.data_test = values[i_test_start:]

        self.len_train = len(self.data_train)
        self.len_val = len(self.data_val)
        self.len_test = len(self.data_test)

    def get_train_data(
        self, seq_len: int, normalise: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """학습용 (x, y). x: (N, seq_len-1, n_cols), y: (N, 1) — Close 예측."""
        data_x, data_y = [], []
        for i in range(self.len_train - seq_len):
            window = self.data_train[i : i + seq_len]
            if normalise:
                window = _normalise_selected_columns(
                    window, self.cols_to_norm, single_window=True
                )[0]
            x = window[:-1]
            y = window[-1, [0]]
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_val_data(
        self, seq_len: int, normalise: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """검증용 (x, y). 검증 구간이 없으면 빈 배열."""
        n_cols = self.data_train.shape[1]
        if self.len_val < seq_len:
            return (
                np.array([]).reshape(0, seq_len - 1, n_cols),
                np.array([]).reshape(0, 1),
            )
        data_windows = np.array(
            [self.data_val[i : i + seq_len] for i in range(self.len_val - seq_len)],
            dtype=float,
        )
        if normalise:
            data_windows = _normalise_selected_columns(
                data_windows, self.cols_to_norm, single_window=False
            )
        x = data_windows[:, :-1, :]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_test_data(
        self,
        seq_len: int,
        normalise: bool,
        cols_to_norm: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """테스트용 (x, y, y_base). y_base는 역정규화용 첫 시점 Close."""
        cols_to_norm = cols_to_norm if cols_to_norm is not None else self.cols_to_norm
        data_windows = np.array(
            [
                self.data_test[i : i + seq_len]
                for i in range(self.len_test - seq_len)
            ],
            dtype=float,
        )
        y_base = data_windows[:, 0, [0]]
        if normalise:
            data_windows = _normalise_selected_columns(
                data_windows, cols_to_norm, single_window=False
            )
        x = data_windows[:, :-1, :]
        y = data_windows[:, -1, [0]]
        return x, y, y_base

    def generate_train_batch(
        self, seq_len: int, batch_size: int, normalise: bool
    ):
        """학습용 미니배치 제너레이터."""
        i = 0
        while i < self.len_train - seq_len:
            x_batch, y_batch = [], []
            for _ in range(batch_size):
                if i >= self.len_train - seq_len:
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                window = self.data_train[i : i + seq_len]
                if normalise:
                    window = _normalise_selected_columns(
                        window, self.cols_to_norm, single_window=True
                    )[0]
                x_batch.append(window[:-1])
                y_batch.append(window[-1, [0]])
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def get_split_bounds(self) -> Optional[dict]:
        """날짜 기준 분할 시 각 구간의 시작/끝 인덱스(및 날짜). 검증용."""
        if self._date_index is None or len(self._date_index) == 0:
            return None
        n = len(self._date_index)
        i_train = self.len_train
        i_val = self.len_train + self.len_val
        def _at(i, default=None):
            return self._date_index[i] if 0 <= i < n else default
        return {
            "train": (0, i_train, _at(0), _at(i_train - 1)),
            "val": (i_train, i_val, _at(i_train), _at(i_val - 1)),
            "test": (i_val, n, _at(i_val), _at(n - 1)),
        }


# 기존 코드에서 그대로 import 가능하도록 별칭 제공 (선택)
DataLoader = FNSPIDDataset
