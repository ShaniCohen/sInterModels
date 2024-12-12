from os.path import basename, dirname
import pandas as pd
import numpy as np
import os


def order_df(df, first_cols=None, last_cols=None, sort_by=None, ascending=True):
    first_cols = first_cols if first_cols else []
    last_cols = last_cols if last_cols else []
    middle_cols = [c for c in df.columns.values if c not in first_cols + last_cols]
    df = df[first_cols + middle_cols + last_cols]
    return df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True) if sort_by else df


def read_df(file_path, encoding='utf-8'):
    if ".csv" in basename(file_path):
        df = pd.read_csv(file_path, encoding=encoding)
    elif ".txt" in basename(file_path):
        df = pd.read_csv(file_path, delimiter=" ")
    else:
        raise ValueError(f"{basename(file_path)} NOT SUPPORTED - only .csv OR .txt")
    return df


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def write_df(df, file_path, encoding='utf-8', keep_index=False):
    create_dir_if_not_exists(dirname(file_path))
    if ".csv" in basename(file_path):
        with open(file_path, 'w') as handle:
            df.to_csv(handle, encoding=encoding, index=keep_index)
    else:
        raise ValueError(f"{basename(file_path)} NOT SUPPORTED - only .csv .pickle .pkl")
    return df


def split_df_samples(df: pd.DataFrame, ratio: float, seed: int = None) -> (pd.DataFrame, pd.DataFrame):
    assert 0 < ratio < 1, "invalid ratio (should be between 0 to 1)"
    if seed is not None:
        np.random.seed(seed)
    num_samp_1 = int(len(df) * ratio)

    all_idx = np.arange(len(df))
    df_1_idx = sorted(np.random.choice(a=all_idx, size=num_samp_1, replace=False))
    df_2_idx = sorted(set(all_idx) - set(df_1_idx))

    df_1 = df.iloc[df_1_idx, :].reset_index(drop=True)
    df_2 = df.iloc[df_2_idx, :].reset_index(drop=True)
    assert len(df_1) + len(df_2) == len(df), "mismatch"

    return df_1, df_2


def get_logger_config_dict(filename: str = 'default_log', file_level: str = 'INFO', console_level: str = 'DEBUG',
                           loggers: dict = None):
    config_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': '%(asctime)s.%(msecs)03d; %(name)s; %(levelname)s %(message)s',
                'datefmt': "%Y-%m-%d %H:%M:%S",
            },
        },
        'handlers': {
            'file_handler': {
                'level': file_level,
                'formatter': 'simple',
                'class': 'logging.FileHandler',
                'filename': f"{filename}.log"
            },
            'console': {
                'level': console_level,
                'class': 'logging.StreamHandler',
                'formatter': 'simple'
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file_handler"]
        }
    }

    if loggers:
        config_dict['loggers'] = loggers

    return config_dict
