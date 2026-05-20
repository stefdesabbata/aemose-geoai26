"""
SOcio-demographic DAta PREParation utilities

This module provides utilities for working with socio-demographic data
(e.g. from the census) for geodemographic classification.

It includes functions for normalising and analysing columns.

Author: Stef De Sabbata
Date: October 2025
Version: 0.1
License: MIT
"""


import numpy as np
import pandas as pd
from typing import Iterable, Union, Callable, Literal, List, Tuple, Optional
from itertools import combinations
import warnings
import logging


# -------------------------------------------------------------------------
# Inline unit tests helper functions
# -------------------------------------------------------------------------

def _assert_list_almost_equal(
        a:   list, 
        b:   list, 
        tol: float=1e-12
    ) -> None:
    """
    Assert two lists are almost equal, element-wise, within a tolerance.
    Parameters
    ----------
    a : list
        First list.
    b : list
        Second list.
    tol : float, default 1e-12
        Tolerance for numerical comparison.

    Raises
    ------
    AssertionError
        If lists are not almost equal.

    Returns
    -------
    None.
    """
    assert len(a) == len(b), f'Length mismatch: {len(a)} != {len(b)}'
    for i, (x, y) in enumerate(zip(a, b)):
        if isinstance(x, (int, float, np.integer, np.floating)) and isinstance(y, (int, float, np.integer, np.floating)):
            assert abs(x - y) <= tol, f'Index {i}: {x} != {y}'
        else:
            assert x == y, f'Index {i}: {x} != {y}'

def _test_assert_list_almost_equal():
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0 + 1e-13]
    _assert_list_almost_equal(a, a)
    _assert_list_almost_equal(b, b)
    _assert_list_almost_equal(a, b)
    _assert_list_almost_equal(b, a)
    try:
        _assert_list_almost_equal([1.0, 2.0], [1.0, 2.1])
        assert False, 'Expected AssertionError'
    except AssertionError:
        pass
    try:
        _assert_list_almost_equal([1.0, 'a'], [1.0, 'b'])
        assert False, 'Expected AssertionError'
    except AssertionError:
        pass
    try:
        _assert_list_almost_equal([1.0, 2.0], [1.0])
        assert False, 'Expected AssertionError'
    except AssertionError:
        pass

# -------------------------------------------------------------------------
# Checking columns helper functions
# -------------------------------------------------------------------------

def _check_no_rows(
        df: pd.DataFrame
    ) -> bool:
    """
    Check if the dataframe has no rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns
    -------
    bool
        True if the dataframe has no rows, False otherwise.
    """
    return df.shape[0] == 0

# Unit tests

def _test_check_no_rows():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [], 'B': []})
    assert not _check_no_rows(df1)
    assert _check_no_rows(df2)

# -------------------------------------------------------------------------

def _check_columns_missing(
        df:      pd.DataFrame,
        columns: Iterable[str]
    ) -> List[str]:
    """
    Return listed columns which are missing from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str]
        Column names to check.
    
    Returns
    -------
    List[str]
        List of missing column names.
    """
    return [c for c in columns if c not in df.columns]

# Unit tests

def _test_check_columns_missing():
    df = pd.DataFrame({
        'A': [1, 2], 
        'B': [3, 4]})
    assert _check_columns_missing(df, ['A', 'B']) == []
    assert _check_columns_missing(df, ['A', 'C']) == ['C']
    assert _check_columns_missing(df, ['C', 'D']) == ['C', 'D']

# -------------------------------------------------------------------------

def _check_columns_nonnumeric(
        df:      pd.DataFrame,
        columns: Iterable[str]
    ) -> List[str]:
    """
    Return listed columns which are non-numeric (including boolean).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str]
        Column names to check.
    
    Returns
    -------
    List[str]
        List of non-numeric (including boolean) column names.
    """
    return [c for c in columns if (not pd.api.types.is_numeric_dtype(df[c])) or pd.api.types.is_bool_dtype(df[c])]

# Unit tests

def _test_check_columns_nonnumeric():
    df = pd.DataFrame({
        'A': [1,    2    ], 
        'B': [3,    4    ], 
        'C': ['s',  't'  ], 
        'D': [True, False], 
        'E': ['r',  'i'  ]})
    assert _check_columns_nonnumeric(df, ['A']) == []
    assert _check_columns_nonnumeric(df, ['A', 'B']) == []
    assert _check_columns_nonnumeric(df, ['A', 'C']) == ['C']
    assert _check_columns_nonnumeric(df, ['A', 'D']) == ['D']
    assert _check_columns_nonnumeric(df, ['C', 'D', 'E']) == ['C', 'D', 'E']

# -------------------------------------------------------------------------

def _check_columns_withnans(
        df:      pd.DataFrame,
        columns: Iterable[str]
    ) -> List[str]:
    """
    Return listed columns which contain NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str]
        Column names to check.
    
    Returns
    -------
    List[str]
        List of column names containing NaNs.
    """
    return [c for c in columns if df[c].isna().any()]

# Unit tests

def _test_check_columns_withnans():
    df = pd.DataFrame({
        'A': [1,    float('nan')], 
        'B': [1,    2           ], 
        'C': [True, False       ]})
    assert _check_columns_withnans(df, ['A', 'B']) == ['A']
    assert _check_columns_withnans(df, ['A', 'C']) == ['A']
    assert _check_columns_withnans(df, ['B', 'C']) == []

# -------------------------------------------------------------------------

def _check_columns_withnegatives(
        df:      pd.DataFrame,
        columns: Iterable[str]
    ) -> List[str]:
    """
    Return listed columns which contain negative values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str]
        Column names to check.
    
    Returns
    -------
    List[str]
        List of column names containing negative values.
    """
    return [c for c in columns if (df[c] < 0).any()]

# Unit tests

def _test_check_columns_withnegatives():
    df = pd.DataFrame({
        'A': [1,        3], 
        'B': [1,       -2], 
        'C': [True, False]})
    assert _check_columns_withnegatives(df, ['A', 'B']) == ['B']
    assert _check_columns_withnegatives(df, ['A', 'C']) == []
    assert _check_columns_withnegatives(df, ['B', 'C']) == ['B']

# -------------------------------------------------------------------------

def _check_columns_withzerosrange(
        df:      pd.DataFrame,
        columns: Iterable[str]
    ) -> List[str]:
    """
    Return listed columns which have zero range (max == min).
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str]
        Column names to check.
    Returns
    -------
    List[str]
        List of column names with zero range.
    """
    return [c for c in columns if df[c].max() == df[c].min()]

# Unit tests

def _test_check_columns_withzerosrange():
    df = pd.DataFrame({
        'A': [1, 1, 1], 
        'B': [1, 2, 3], 
        'C': [0, 0, 0]})
    assert _check_columns_withzerosrange(df, ['A', 'B']) == ['A']
    assert _check_columns_withzerosrange(df, ['A', 'C']) == ['A', 'C']
    assert _check_columns_withzerosrange(df, ['B', 'C']) == ['C']
    assert _check_columns_withzerosrange(df, ['B']) == []

# -------------------------------------------------------------------------

def _check_all(
        df:      pd.DataFrame,
        columns: Iterable[str]
    ) -> None:
    """
    Run all checks on the specified dataframe: 
    - non-zero number of rows;
    - specified columns exist;
    - specified columns have numeric type (not boolean);
    - specified columns have no NaNs;
    - specified columns have non-zero range.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str]
        Column names to check.
    
    Returns
    -------
    None.
    
    Raises
    ------
    KeyError
        If any column is missing.
    TypeError
        If any column is not numeric.
    ValueError
        If any column contains NaN or negative values, or no rows at all.
    """
    # Check dataframe has rows
    if _check_no_rows(df):
        raise ValueError('DataFrame contains no rows.')

    # Check all requested columns exist
    cols_missing = _check_columns_missing(df, columns)
    if cols_missing:
        raise KeyError(f'Columns must be in the dataframe. Please check the following columns exist: {cols_missing}.')
    
    # Ensure columns are numeric
    cols_nonnum = _check_columns_nonnumeric(df, columns)
    if cols_nonnum:
        raise TypeError(f'Columns must be numeric. Please convert or drop: {cols_nonnum}.')

    # Check for NaN values in the columns
    cols_wthnan = _check_columns_withnans(df, columns)
    if cols_wthnan:
        raise ValueError(f'Columns contain NaN values. Please handle NaNs in the following columns: {cols_wthnan}.')

    # Check for negative values in the columns
    cols_wngtvs = _check_columns_withnegatives(df, columns)
    if cols_wngtvs:
        raise ValueError(f'Columns contain negative values. Please handle negative values in the following columns: {cols_wngtvs}.')
    
    # Check for zero range in the columns
    cols_wzrnge = _check_columns_withzerosrange(df, columns)
    if cols_wzrnge:
        raise ValueError(f'Columns have zero range (max == min). Please check these columns: {cols_wzrnge}.')

    return None


# -------------------------------------------------------------------------
# Other helper functions
# -------------------------------------------------------------------------

def _new_column_name(
        column_name: str,
        suffix:      Union[str, bool, None] = '_new'
    ) -> str:
    """
    Determine suffix for new column name based on suffix parameter.

    Parameters
    ----------
    column_name : str
        Original column name.
    suffix : str or bool, default '_new'
        Suffix to append to the new column name. 
        Column is replaced if empty string, None or False is provided.
    
    Returns
    -------
    str
        Suffix for new column names.
    """
    if suffix in [None, '', False]:
        new_column_name = f'{column_name}'
    elif isinstance(suffix, str):
        new_column_name = f'{column_name}{suffix}'
    elif suffix is True:
        new_column_name = f'{column_name}_new'
    else:
        new_column_name = f'{column_name}_{str(suffix)}'
    return new_column_name
    
# Unit tests

def _test_new_column_name():
    assert _new_column_name('A')         == 'A_new'
    assert _new_column_name('A', True)   == 'A_new'
    assert _new_column_name('A', '')     == 'A'
    assert _new_column_name('A', None)   == 'A'
    assert _new_column_name('A', False)  == 'A'
    assert _new_column_name('A', '_out') == 'A_out'
    assert _new_column_name('A', 123)    == 'A_123'


# -------------------------------------------------------------------------
# Utils functions: normalisation
# -------------------------------------------------------------------------

def _get_col_by_total(
        df:             pd.DataFrame,
        column_target:  str,
        column_total:   str,
        clip:           bool = False,
        to_percent:     bool = False,
        warn:           bool = True
    ) -> np.ndarray:
    """
    Returns a numpy array containing the values of the target column using totals 
    from another column. This function is designed for cases where the target column 
    lists counts of a subset of the total in the totals column, e.g. number of people 
    aged 25-34 (target) as a proportion of total population (total).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column_target : str
        Column name to normalise.
    column_total : str
        Column name containing totals for normalisation.
    clip : bool, default False
        If True, clip the result to the ``[0.0, 1.0]`` range.
    to_percent : bool, default False
        If True, return outputs as percentages (0.0-100.0) rather than unit range (0.0-1.0).
    warn : bool, default True
        Warn when the computed result falls outside ``[0.0, 1.0]``.
    
    Returns
    -------
    np.ndarray
        Array containing the normalised target column.
    
    Raises
    ------
    KeyError
        If either column is missing.
    TypeError
        If either column is not numeric.
    ValueError
        If either column contains NaN or negative values, or no rows at all.
    """
    
    # --- Normalisation ---
    
    target = df[column_target].to_numpy(dtype=float)
    total  = df[column_total ].to_numpy(dtype=float)

    # Warn about presence of zero totals
    num_zero_totals = np.sum(total == 0)
    if num_zero_totals > 0:
        warnings.warn(
            f'({column_target}/{column_total}) {num_zero_totals} rows have the value zero listed as total. The normalised value will be set to zero for these rows.',
            UserWarning
        )

    # Compute normalisation by total, handling zero totals
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(
            target, total, 
            # set result to 0.0 where total is zero
            out=np.zeros_like(target, dtype=float), 
            # function output where total is not zero
            where=total != 0)

    # Convert to percentage if requested
    factor  = 1.0
    if to_percent:
        factor  = 100.0
        result *= factor

    # Clip or count out-of-range values
    if clip:
        clipped            = np.clip(result, 0.0, factor)
        out_of_range_count = np.sum(clipped != result)
        result             = clipped
    else:
        out_of_range_count = np.sum((result < 0.0) | (result > factor))

    # Warnings about values outside range
    if warn:
        if out_of_range_count > 0:
            warnings.warn(
                f'({column_target}/{column_total}) {out_of_range_count} rows resulted in values outside the [0.0, {factor:.1f}] range.',
                UserWarning
            )
    
    return result

# Unit tests

def _test_get_col_by_total():
    a_vals = [  0,  1,      2,   3,     4,   5  ]
    b_vals = [  4,  4,      2,   0,     2,   0  ]
    r_vals = [0.0,  0.25,   1.0, 0.0,   2.0, 0.0]
    c_vals = [0.0,  0.25,   1.0, 0.0,   1.0, 0.0]
    p_vals = [0.0, 25.0,  100.0, 0.0, 200.0, 0.0]
    q_vals = [0.0, 25.0,  100.0, 0.0, 100.0, 0.0]
    df = pd.DataFrame({
        'A': a_vals, 
        'B': b_vals})
    # Base
    with warnings.catch_warnings(record=True) as w:
        out = _get_col_by_total( df, 'A', 'B')
        _assert_list_almost_equal(out.tolist(), r_vals)
        assert len(w) == 2, 'Expected a warning to be raised'
        assert any(issubclass(wi.category, UserWarning) for wi in w), 'Expected a UserWarning'
        for warn in w:
            print(f'[Expected warning in _test_get_col_by_total] {warn.message}')
    # Clip
    with warnings.catch_warnings(record=True) as w:
        out = _get_col_by_total( df, 'A', 'B', clip=True)
        _assert_list_almost_equal(out.tolist(), c_vals)
        assert len(w) == 2, 'Expected a warning to be raised'
        assert any(issubclass(wi.category, UserWarning) for wi in w), 'Expected a UserWarning'
        for warn in w:
            print(f'[Expected warning in _test_get_col_by_total] {warn.message}')
    # To percentage
    with warnings.catch_warnings(record=True) as w:
        out = _get_col_by_total( df, 'A', 'B', to_percent=True)
        _assert_list_almost_equal(out.tolist(), p_vals)
        assert len(w) == 2, 'Expected a warning to be raised'
        assert any(issubclass(wi.category, UserWarning) for wi in w), 'Expected a UserWarning'
        for warn in w:
            print(f'[Expected warning in _test_get_col_by_total] {warn.message}')
    # To percentage and clip
    with warnings.catch_warnings(record=True) as w:
        out = _get_col_by_total( df, 'A', 'B', to_percent=True, clip=True)
        _assert_list_almost_equal(out.tolist(), q_vals)
        assert len(w) == 2, 'Expected a warning to be raised'
        assert any(issubclass(wi.category, UserWarning) for wi in w), 'Expected a UserWarning'
        for warn in w:
            print(f'[Expected warning in _test_get_col_by_total] {warn.message}')


# -------------------------------------------------------------------------

def get_normd_by_total(
    df:            pd.DataFrame,
    col_tot_pairs: Iterable[Tuple[str, str]],
    clip:          bool = False,
    to_percent:    bool = False,
    suffix:        str = '_ntot',
    warn:          bool = True
    ) -> pd.DataFrame:
    """
    Returns a new dataframe containing only the normalised values for the target columns.
    The function normalises multiple target columns using their corresponding totals columns.
    This function is designed for cases where target columns list counts of subsets
    of totals in the corresponding totals columns, e.g. number of people aged 25-34 (target) 
    as a proportion of total population (total). The suffix '_ntot' is added to the original
    column names by default.
    The input data columns are expected to be non-empty, numeric, non-negative, having 
    non-zero range and without missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col_tot_pairs : Iterable[Tuple[str, str]]
        Pairs of (column_target, column_total) where:
            - column_target: Column name to normalise
            - column_total: Column name containing totals for normalisation
    clip : bool, default False
        If True, clip the result to the ``[0.0, 1.0]`` range.
    to_percent : bool, default False
        If True, return outputs as percentages (0.0-100.0) rather than unit range (0.0-1.0).
    suffix : str, default '_ntot'
        Suffix to append to the new column name.
    warn : bool, default True
        Warn when the computed result falls outside ``[0.0, 1.0]``.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with normalised target columns (copy).
    
    Raises
    ------
    KeyError
        If any column is missing.
    TypeError
        If any column is not numeric.
    ValueError
        If any column contains NaN or negative values, or no rows at all.
    """

    # Work on a copy of the dataframe for safety
    df = df.copy(deep=True)

    # --- Checks ---
    # Get all columns from pairs
    all_cols_for_check = set()
    for column_target, column_total in col_tot_pairs:
        all_cols_for_check.add(column_target)
        all_cols_for_check.add(column_total)
    all_cols_for_check = list(all_cols_for_check)
    # Run checks
    _check_all(df, all_cols_for_check)

    # --- Normalisation ---

    norm_cols = {}
    for col_target, col_total in col_tot_pairs:
        # Apply normalisation
        norm_vals = _get_col_by_total(
            df, 
            column_target=col_target, 
            column_total=col_total,
            clip=clip,
            to_percent=to_percent,
            warn=warn
        )
        norm_cols[_new_column_name(col_target, suffix)] = norm_vals
    
    # Create output dataframe
    norm_cols_df = pd.DataFrame(norm_cols, index=df.index)    
    return norm_cols_df
    
def _test_get_normd_by_total():
    a_vals  = [0,    1,      2,    3,     4,    5  ]
    b_vals  = [0,   10,     20,   30,    40,   50  ]
    n_vals  = [1,   11,     21,   31,    41,   51  ]
    t_vals  = [4,   40,     20,    0,    20,    0  ]
    x0_vals = [0.0,  0.025,  0.1,  0.0,   0.2,  0.0]
    y0_vals = [0.0,  0.25,   1.0,  0.0,   2.0,  0.0]
    xp_vals = [0.0,  2.5,   10.0,  0.0,  20.0,  0.0]
    yp_vals = [0.0, 25,    100.0,  0.0, 200.0,  0.0]
    xc_vals = [0.0,  0.025,  0.1,  0.0,   0.2,  0.0]
    yc_vals = [0.0,  0.25,   1.0,  0.0,   1.0,  0.0]
    xq_vals = [0.0,  2.5,   10.0,  0.0,  20.0,  0.0]
    yq_vals = [0.0, 25.0,  100.0,  0.0, 100.0,  0.0]
    df = pd.DataFrame({
        'A': a_vals, 
        'B': b_vals,
        'N': n_vals, 
        'T': t_vals})
    cols_tots = [('A', 'T'), ('B', 'T')]
    out = get_normd_by_total(df, cols_tots)
    assert len(out.columns) == 2, 'Expected a warning to be raised'
    _assert_list_almost_equal(out['A_ntot'].tolist(), x0_vals)
    _assert_list_almost_equal(out['B_ntot'].tolist(), y0_vals)
    out = get_normd_by_total(df, cols_tots, to_percent=True)
    assert len(out.columns) == 2, 'Expected a warning to be raised'
    _assert_list_almost_equal(out['A_ntot'].tolist(), xp_vals)
    _assert_list_almost_equal(out['B_ntot'].tolist(), yp_vals)
    out = get_normd_by_total(df, cols_tots, clip=True)
    assert len(out.columns) == 2, 'Expected a warning to be raised'
    _assert_list_almost_equal(out['A_ntot'].tolist(), xc_vals)
    _assert_list_almost_equal(out['B_ntot'].tolist(), yc_vals)
    out = get_normd_by_total(df, cols_tots, to_percent=True, clip=True)
    assert len(out.columns) == 2, 'Expected a warning to be raised'
    _assert_list_almost_equal(out['A_ntot'].tolist(), xq_vals)
    _assert_list_almost_equal(out['B_ntot'].tolist(), yq_vals)

# -------------------------------------------------------------------------

def get_normd_min_max(
        df:         pd.DataFrame,
        to_percent: bool = False,
        suffix:     str = '_minmax',
    ) -> pd.DataFrame:
    """
    Returns a new dataframe containing only the normalised values for the target columns.
    The function applies the min-max normalisation to all columns, using the formula: 
    (x - min) / (max - min).
    The suffix '_minmax' is added to the original column names by default.
    The input data columns are expected to be non-empty, numeric, non-negative, having 
    non-zero range and without missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    to_percent : bool, default False
        If True, return outputs as percentages (0.0-100.0) rather than unit range (0.0-1.0).
    suffix : str, default '_minmax'
        Suffix to append to the new column name. Column is replaced if empty string, None or False is provided.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with normalised columns.
    
    Raises
    ------
    KeyError
        If any column is missing.
    TypeError
        If any column is not numeric.
    ValueError
        If any column contains NaN or negative values, or no rows at all, or if max == min.
    """

    # Work on a copy of the dataframe for safety
    df = df.copy(deep=True)

    # Get all columns
    columns = df.columns.tolist()

    # --- Checks ---
    _check_all(df, columns)

    # --- Normalisation ---

    norm_cols = {}
    # Apply min-max normalisation
    for col in columns:
        # Calculate min and max
        col_min = df[col].min()
        col_max = df[col].max()
        col_rng = col_max - col_min
        # Apply min-max normalisation
        norm_vals = (df[col].to_numpy(dtype=float) - col_min) / (col_rng)
        if to_percent:
            norm_vals *= 100.0
        norm_cols[_new_column_name(col, suffix)] = norm_vals

    # Create output dataframe
    norm_cols_df = pd.DataFrame(norm_cols, index=df.index)    
    return norm_cols_df

# Unit tests

def _test_get_normd_min_max():
    df = pd.DataFrame({
        'A': [ 0,  5, 10], 
        'B': [ 9,  3,  3]})
    # Base
    out = get_normd_min_max(df)
    _assert_list_almost_equal(out['A_minmax'].tolist(), [0.0, 0.5, 1.0])
    _assert_list_almost_equal(out['B_minmax'].tolist(), [1.0, 0.0, 0.0])
    # To percentage
    out = get_normd_min_max(df, to_percent=True)
    _assert_list_almost_equal(out['A_minmax'].tolist(), [  0.0, 50.0, 100.0])
    _assert_list_almost_equal(out['B_minmax'].tolist(), [100.0,  0.0,   0.0])


# -------------------------------------------------------------------------
# Utils functions: outlier handling
# -------------------------------------------------------------------------

def get_clipped_outliers(
        df:             pd.DataFrame,
        columns:        Optional[Iterable[str]],
        method:         Literal['iqr', 'iqr_diff', 'zscore'] = 'iqr',
        factor:         float = 3.0,
        suffix:         str = '_clipd',
        suffix_noout:   str = '_noout',
        warn:           bool = True
    ) -> pd.DataFrame:
    """
    Returns a new dataframe containing only the clipped values for the target columns.
    The function identifies extreme outliers in specified columns (all columns if columns is None) 
    and clips them to the minimum and maximum values calculated excluding those outliers.
    The suffix '_clipd' is added by default to the original column names where outliers 
    have been clipped and '_noout' where none have been found.
    The input data columns are expected to be non-empty, numeric, non-negative, having 
    non-zero range and without missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str], default None
        Column names to process.
    method : {'iqr', 'iqr_diff', 'zscore'}, default 'iqr'
        Method to identify outliers:
        - 'iqr': Values outside Q3 + factor*IQR or below Q1 - factor*IQR
        - 'iqr_diff': As 'iqr' but using more extreme percentiles to find
        "safe" Q1 and Q3 bounds to avoid Q1 == Q3 in extreme distributions.
        - 'zscore': Values with absolute z-score greater than factor
    factor : float, default 3.0
        Multiplier used in outlier detection (e.g., 3.0 for z-score means 
        values beyond 3 standard deviations).
    suffix : str, default '_clipd'
        Suffix to append to the new column name, where outliers have been clipped.
    suffix_noout : str, default '_noout'
        Suffix to append to the new column name, where no outliers were found.
    warn : bool, default True
        Warn when any values are clipped.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with clipped outliers.
    
    Raises
    ------
    KeyError
        If any column is missing.
    TypeError
        If any column is not numeric.
    ValueError
        If any column contains NaN or negative values, or no rows at all, or if method is invalid.
    """
    # Work on a copy of the dataframe for safety
    df = df.copy(deep=True)

    # Input validation
    if method not in ['iqr', 'iqr_diff', 'zscore']:
        raise ValueError('Method must be either \'iqr\', \'iqr_diff\' or \'zscore\'.')
    if factor <= 0:
        raise ValueError('Factor must be a positive number')
    
    if columns is None:
        columns = df.columns.tolist()
    
    # --- Checks ---
    _check_all(df, columns)
    
    # -- Clip outliers ---

    clip_cols = {}
    for col in columns:
        
        values = df[col].to_numpy(dtype=float)
        
        if method == 'iqr' or method == 'iqr_diff':
            # Interquartile range (IQR)
            # Canonical Q1 and Q3
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            if method == 'iqr_diff' and q1 == q3:
                # Find safe bounds
                q1s  = [(p, np.percentile(values, p, method='closest_observation')) for p in range(1, 26)]
                q3s  = [(p, np.percentile(values, p, method='closest_observation')) for p in range(75, 100)]
                q1s  = [q for q in q1s if q[1] < q3]
                q3s  = [q for q in q3s if q[1] > q1]
                if q1s:
                    if warn:
                        warnings.warn(
                            f'({col}) Using {q1s[-1][0]} percentile = {q1s[-1][1]:.3f} instead of Q1 (25 percentile) to avoid IQR==0.',
                            UserWarning
                        )
                    q1 = q1s[-1][1] 
                else:
                    if warn:
                        warnings.warn(
                            f'({col}) Using min value = {np.min(values):.3f} instead of Q1 (25 percentile) to avoid IQR==0.',
                            UserWarning
                        )
                    q1 = np.min(values)
                if q3s:
                    if warn:
                        warnings.warn(
                            f'({col}) Using {q3s[0][0]} percentile = {q3s[0][1]:.3f} instead of Q3 (75 percentile) to avoid IQR==0.',
                            UserWarning
                        )
                    q3 = q3s[0][1] 
                else:
                    if warn:
                        warnings.warn(
                            f'({col}) Using max value = {np.max(values):.3f} instead of Q3 (75 percentile) to avoid IQR==0.',
                            UserWarning
                        )
                    q3 = np.max(values)
            iqr = q3 - q1
            # Bounds
            lower_bound = max((q1 - (factor * iqr)), np.min(values))
            upper_bound = min((q3 + (factor * iqr)), np.max(values))
            # Outliers
            outliers_mask = (values < lower_bound) | (values > upper_bound)

        else:
            # Z-scores
            mean = np.mean(values)
            std  = np.std(values)
            if std < 1e-12:
                raise ValueError(f'({col}) Standard deviation is lower than 1e-12 for column {col}. Cannot compute z-scores.')
            z_scores_abs  = np.abs((values - mean) / std)
            # Outliers
            outliers_mask = z_scores_abs > factor
        
        # If no outliers found, return original column
        if not np.any(outliers_mask):
            if warn:
                warnings.warn(
                    f'({col}) No values marked as outliers for column. Returning original values as column {_new_column_name(col, suffix_noout)}.',
                    UserWarning
                )
            
            # Set result to orginal values
            clip_cols[_new_column_name(col, suffix_noout)] = values

        else:            
            # Calculate new maximum and minimum excluding outliers
            non_outliers = values[~outliers_mask]
            if len(non_outliers) == 0:
                raise ValueError(f'({col}) All values marked as outliers for column. Please check.')
            
            max_non_outlier = np.max(non_outliers)
            min_non_outlier = np.min(non_outliers)

            # Clip outliers
            clipped = np.clip(values, min_non_outlier, max_non_outlier)
            if warn:
                warnings.warn(
                    f'({col}) {len(values[outliers_mask])} outliers have been clipped to bounds: {min_non_outlier:.3f}, {max_non_outlier:.3f} [{method}, {factor}].',
                    UserWarning
                )

            # Set result to clipped values
            clip_cols[_new_column_name(col, suffix)] = clipped

    # Create clipped dataframe
    clip_cols_df = pd.DataFrame(clip_cols, index=df.index)
    return clip_cols_df

# Unit tests

def _test_get_clipped_outliers_iqr():
    with warnings.catch_warnings(record=True) as w:
        df = pd.DataFrame({
            'A': [  1, 2, 3, 4, 5, 20],
            'B': [100, 2, 3, 4, 5,  6],
            'C': [  2, 3, 4, 5, 6,  7]
        })
        out = get_clipped_outliers(df, ['A', 'B', 'C'])
        assert len(w) == 3, 'Expected a warning to be raised'
        assert any(issubclass(wi.category, UserWarning) for wi in w), 'Expected a UserWarning'
        for warn in w:
            print(f'[Expected warning in _test_get_clipped_outliers_iqr] {warn.message}')
        assert out['A_clipd'].tolist() == [1, 2, 3, 4, 5, 5]
        assert out['B_clipd'].tolist() == [6, 2, 3, 4, 5, 6]
        assert out['C_noout'].tolist() == df['C'].tolist()
        assert len(out.columns) == 3, 'Expected 3 columns in output'

def _test_get_clipped_outliers_zscore():
    with warnings.catch_warnings(record=True) as w:
        df = pd.DataFrame({
            'A': [2, 3, 4, 5, 6, 20],
            'B': [1.5, 1002, 1003, 1004, 1005, 1006],
            'C': [2, 3, 4, 5, 6, 7]
        })
        out = get_clipped_outliers(df, ['A', 'B', 'C'], factor=2.0, method='zscore')
        assert len(w) == 3, 'Expected a warning to be raised'
        assert any(issubclass(wi.category, UserWarning) for wi in w), 'Expected a UserWarning'
        for warn in w:
            print(f'[Expected warning in _test_get_clipped_outliers_zscore] {warn.message}')
        assert out['A_clipd'].tolist() == [2, 3, 4, 5, 6, 6]
        assert out['B_clipd'].tolist() == [1002, 1002, 1003, 1004, 1005, 1006]
        assert out['C_noout'].tolist() == df['C'].tolist()
        assert len(out.columns) == 3, 'Expected 3 columns in output'

def _test_get_clipped_outliers_diff():
    with warnings.catch_warnings(record=True) as w:
        df = pd.DataFrame({
            'A': [1, 99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        })
        out = get_clipped_outliers(df, ['A'], method='iqr')
        assert len(w) == 1, 'Expected a warning to be raised'
        assert any(issubclass(wi.category, UserWarning) for wi in w), 'Expected a UserWarning'
        for warn in w:
            print(f'[Expected warning in _test_get_clipped_outliers_diff] {warn.message}')
        assert out['A_clipd'].tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    with warnings.catch_warnings(record=True) as w:
        df = pd.DataFrame({
            'A': [1, 99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        })
        out = get_clipped_outliers(df, ['A'], method='iqr_diff')
        assert len(w) == 3, 'Expected a warning to be raised'
        assert any(issubclass(wi.category, UserWarning) for wi in w), 'Expected a UserWarning'
        for warn in w:
            print(f'[Expected warning in _test_get_clipped_outliers_diff] {warn.message}')
        assert out['A_clipd'].tolist() == [ 99,  99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]


# -------------------------------------------------------------------------
# Utils functions: transformation
# -------------------------------------------------------------------------

def get_transformed_ihs(
        df:             pd.DataFrame,
        columns:        Optional[Iterable[str]],
        suffix:         str = '_ihs'
    ) -> pd.DataFrame:
    """
    Returns a new dataframe containing only the transformed values for the target columns.
    The function applies the inverse hyperbolic sine transformation to the specified columns 
    (all columns if columns is None).
    The suffix '_ihs' is added by default to the original column names.
    The input data columns are expected to be non-empty, numeric, non-negative, having 
    non-zero range and without missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str], default None
        Column names to transform.
    suffix : str, default '_ihs'
        Suffix to append to the new column name.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with transformed columns.
    
    Raises
    ------
    KeyError
        If any column is missing.
    TypeError
        If any column is not numeric.
    ValueError
        If any column contains NaN or negative values, or no rows at all.
    """
    # Work on a copy of the dataframe for safety
    df = df.copy(deep=True)

    # Get all columns if none specified
    if columns is None:
        columns = df.columns.tolist()

    # --- Checks ---
    _check_all(df, columns)

    # --- Apply transformation ---

    ihs_cols = {}
    for col in columns:
        # Apply inverse hyperbolic sine transformation
        ihs_cols[
                _new_column_name(col, suffix)
            ] = np.arcsinh(
                df[col].to_numpy(dtype=float)
            )
    
    # Create transformed dataframe
    ihs_cols_df = pd.DataFrame(ihs_cols, index=df.index)

    return ihs_cols_df

# Unit tests

def _test_get_transformed_ihs():
    df = pd.DataFrame({
        'A': [0, 1, 2, 10, 100],
        'B': [2, 3, 4,  5,   6],
        'C': ['x', 'y', 'z', 'a', 'b']
    })
    out = get_transformed_ihs(df, ['A', 'B'])
    expected_A = np.arcsinh([0, 1, 2, 10, 100]).tolist()
    expected_B = np.arcsinh([2, 3, 4,  5,   6]).tolist()
    _assert_list_almost_equal(out['A_ihs'].tolist(), expected_A)
    _assert_list_almost_equal(out['B_ihs'].tolist(), expected_B)
    assert 'C_ihs' not in out.columns
    assert 'C'     not in out.columns


# -------------------------------------------------------------------------
# Utils functions: correlation
# -------------------------------------------------------------------------

def get_correlated_pairs(
        df:           pd.DataFrame,
        columns:      Optional[Iterable[str]],
        threshold:    float = 0.0,
        method:       Union[
            Literal['pearson', 'kendall', 'spearman'], 
            Callable[[np.ndarray, np.ndarray], float]
            ]='pearson', 
        min_periods:  int=1, 
        numeric_only: bool=False
    ) -> List[dict]:
    """
    Find pairs of columns among those specified (all columns if columns is None)
    with correlation at or above a given threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str], default None
        Column names to check for correlation.
    threshold : float
        The absolute correlation value threshold.
    method : {'pearson', 'kendall', 'spearman'} or callable, default 'pearson'
        Method to compute correlation. See `pandas.DataFrame.corr` for details.
    min_periods : int, default 1
        Minimum number of observations required per pair of columns to have a valid result.
        See `pandas.DataFrame.corr` for details.
    numeric_only : bool, default False
        If True, include only float, int, boolean columns. See `pandas.DataFrame.corr` for details.

    Returns
    -------
    List[dict]
        A list of dictionaries, where each dictionary contains two column names
        and their correlation value: {column_x_name, column_y_name, correlation}.
    
    Raises
    ------
    KeyError
        If any column is missing.
    TypeError
        If any column is not numeric.
    ValueError
        If any column contains NaN or negative values, or no rows at all.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError('Threshold must be between 0 and 1 (inclusive).')
    
    if columns is None:
        columns = df.columns.tolist()
    
    # --- Checks ---

    _check_all(df, columns)

    # --- Correlation analysis ---

    # If there are fewer than 2 columns, return an empty list
    if len(columns) < 2:
        return []
    
    # Calculate correlations
    corr_matrix = df[columns].corr(method=method, min_periods=min_periods, numeric_only=numeric_only)

    # Create correlation pairs list
    correlated_pairs = []
    for col1, col2 in combinations(columns, 2):
        correlation = corr_matrix.loc[col1, col2]
        if abs(correlation) >= threshold:
            correlated_pairs.append({
                'column_x_name': col1, 
                'column_y_name': col2, 
                'correlation':   correlation})
            
    return correlated_pairs

# Unit tests

def _test_get_correlated_pairs_basic():
    df = pd.DataFrame({
        'A': [ 1,    2,    3,    4,     5 ],
        'B': [ 2,    4,    6,    8,    10 ],
        'C': [19,   18,   17,   16,    15 ],
        'D': [ 5,    1,    4,    2,     3 ],
        'E': [ 1.1,  1.9,  3.2,  1,     4.9]
    })
    
    # Test high threshold
    pairs = get_correlated_pairs(df, ['A', 'B', 'C', 'D', 'E'], 0.99)
    # print(pairs)
    assert len(pairs) == 3
    _assert_list_almost_equal(list(pairs[0].values()), ['A', 'B', 1.0])
    _assert_list_almost_equal(list(pairs[1].values()), ['A', 'C', -1.0])
    _assert_list_almost_equal(list(pairs[2].values()), ['B', 'C', -1.0])

    # Test lower threshold
    pairs_lower = get_correlated_pairs(df, ['A', 'B', 'C', 'D', 'E'], 0.5)
    # print(pairs_lower)
    assert len(pairs_lower) == 6
    assert set([c for p in pairs_lower for c in (p['column_x_name'], p['column_y_name'])]) == {'A', 'B', 'C', 'E'}

def _test_get_correlated_pairs_empty():
    df = pd.DataFrame({
        'A': [1, 2], 
        'B': [3, 4]})
    assert get_correlated_pairs(df, [], 0.5) == []
    assert get_correlated_pairs(df, ['A'], 0.5) == []

def _test_get_correlated_pairs_threshold():
    df = pd.DataFrame({
        'A': [1, 2, 3], 
        'B': [4, 5, 6]})
    try:
        get_correlated_pairs(df, ['A', 'B'], -0.1)
        assert False, 'Expected ValueError'
    except ValueError:
        pass
    try:
        get_correlated_pairs(df, ['A', 'B'], 1.1)
        assert False, 'Expected ValueError'
    except ValueError:
        pass

# -------------------------------------------------------------------------
# Utils functions: variance
# -------------------------------------------------------------------------

def get_low_variance_cols(
        df:        pd.DataFrame,
        columns:   Optional[Iterable[str]],
        threshold: float = 0.0
    ) -> List[dict]:
    """
    Find columns among those specified (all columns if columns is None) 
    with variance at or below a given threshold, and count number of zeros.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str], default None
        Column names to check for variance.
    threshold : float
        The variance threshold. Columns with variance below this will be returned.

    Returns
    -------
    List[dict]
        A list of dictionaries, where each dictionary contains a column name, its variance value
        and the count of cases which are equal to zero: {column_name, variance, num_of_zeros}.
    
    Raises
    ------
    KeyError
        If any column is missing.
    TypeError
        If any column is not numeric.
    ValueError
        If any column contains NaN or negative values, or no rows at all.
    """
    if threshold < 0:
        raise ValueError('Threshold must be non-negative.')
    
    if columns is None:
        columns = df.columns.tolist()
    
    # --- Checks ---

    _check_all(df, columns)

    # --- Variance analysis ---
    
    low_var_columns = []
    for col in columns:
        variance =  df[col].var()
        zeros    = (df[col] == 0).sum()
        if variance <= threshold:
            low_var_columns.append({
                'column_name':  col, 
                'variance':     variance, 
                'num_of_zeros': zeros})
            
    return low_var_columns

# Unit tests

def _test_get_low_variance_cols_basic():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4,  5  ],
        'B': [1, 1, 1, 1,  1.1],
        'C': [2, 3, 5, 7, 11  ],
        'D': [1, 2, 4, 8, 16  ],
        'E': [0, 0, 0, 0,  1  ]
    })
    assert get_low_variance_cols(df, ['A', 'C'], 0.1) == []

    low_var_cols = get_low_variance_cols(df, ['A', 'B', 'C', 'D'], 0.1)
    assert len(low_var_cols) == 1
    _assert_list_almost_equal(list(low_var_cols[0].values()), ['B', 0.002, 0])
    
    low_var_cols = get_low_variance_cols(df, ['B', 'D'], 0.1)
    assert len(low_var_cols) == 1
    _assert_list_almost_equal(list(low_var_cols[0].values()), ['B', 0.002, 0])
    
    low_var_cols = get_low_variance_cols(df, ['A', 'E'], 0.5)
    assert len(low_var_cols) == 1
    _assert_list_almost_equal(list(low_var_cols[0].values()), ['E', 0.2, 4], tol=1e-3)

def _test_get_low_variance_cols_empty():
    df = pd.DataFrame({
        'A': [1, 2], 
        'B': [3, 4]})
    assert get_low_variance_cols(df, [],    0.25) == []
    assert get_low_variance_cols(df, ['A'], 0.25) == []

def _test_get_low_variance_cols_threshold():
    df = pd.DataFrame({
        'A': [1, 2, 3], 
        'B': [4, 5, 6]})
    try:
        get_low_variance_cols(df, ['A', 'B'], -0.1)
        assert False, 'Expected ValueError'
    except ValueError:
        pass

# -------------------------------------------------------------------------

def get_small_range_cols(
        df:        pd.DataFrame,
        columns:   Optional[Iterable[str]],
        quant_bot: float = 25.0,
        quant_top: float = 75.0,
        threshold: float = 0.0
    ) -> List[dict]:
    """
    Find columns among those specified (all columns if columns is None) 
    with small range between the specified quantiles.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Iterable[str], default None
        Column names to check for variance.
    quant_bot : float, default 0.25
        The lower quantile (between 0.0 and 1.0).
    quant_top : float, default 0.75
        The upper quantile (between 0.0 and 1.0).
    threshold : float
        The variance threshold. Columns with variance below this will be returned.

    Returns
    -------
    List[dict]
        A list of dictionaries, where each dictionary contains a column name, its variance value
        and the count of cases which are equal to zero: {column_name, variance, num_of_zeros}.
    
    Raises
    ------
    KeyError
        If any column is missing.
    TypeError
        If any column is not numeric.
    ValueError
        If any column contains NaN or negative values, or no rows at all.
    """
    if threshold < 0:
        raise ValueError('Threshold must be non-negative.')
    
    if columns is None:
        columns = df.columns.tolist()
    
    # --- Checks ---

    _check_all(df, columns)

    # --- Variance analysis ---
    
    small_range_cols = []
    for col in columns:
        qb = np.percentile(df[col], quant_bot)
        qt = np.percentile(df[col], quant_top)
        range =  qt - qb
        if range <= threshold:
            small_range_cols.append({
                'column_name':  col, 
                'quant_bot':    qb,
                'quant_top':    qt,
                'range':     range})
            
    return small_range_cols

# Unit tests

def _test_get_small_range_cols():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4,  5  ],
        'B': [1, 1, 1, 1,  1.1],
        'C': [2, 3, 5, 7, 11  ],
        'D': [1, 2, 4, 8, 16  ],
        'E': [0, 0, 0, 0,  1  ]
    })
    assert get_small_range_cols(df, ['A', 'C'], threshold=1) == []

    small_range_cols = get_small_range_cols(df, ['A', 'B', 'C', 'D'], threshold=1)
    assert len(small_range_cols) == 1
    
    small_range_cols = get_small_range_cols(df, ['B', 'D'], threshold=1)
    assert len(small_range_cols) == 1
    
    small_range_cols = get_small_range_cols(df, ['A', 'E'], threshold=1)
    assert len(small_range_cols) == 1

# -------------------------------------------------------------------------
# Run inline tests if this script is executed directly
# -------------------------------------------------------------------------

# Configure logging
def _simple_warning_format(message, category, filename, lineno, file=None, line=None):
    logging.warning(f'{category.__name__}: {message}')

# Run all tests
def _run_tests():
    # Configure logging and route warnings to the logging framework
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    # logging.captureWarnings(True)
    warnings.showwarning = _simple_warning_format
    warnings.simplefilter('always')

    _test_assert_list_almost_equal()
    _test_new_column_name()

    _test_check_no_rows()
    _test_check_columns_missing()
    _test_check_columns_nonnumeric()
    _test_check_columns_withnans()
    _test_check_columns_withnegatives()
    _test_check_columns_withzerosrange()
    
    _test_get_col_by_total()
    _test_get_normd_by_total()
    
    _test_get_normd_min_max()
    
    _test_get_clipped_outliers_iqr()
    _test_get_clipped_outliers_zscore()
    _test_get_clipped_outliers_diff()

    _test_get_transformed_ihs()
    
    _test_get_correlated_pairs_basic()
    _test_get_correlated_pairs_empty()
    _test_get_correlated_pairs_threshold()
    
    _test_get_low_variance_cols_basic()
    _test_get_low_variance_cols_empty()
    _test_get_low_variance_cols_threshold()
    _test_get_small_range_cols()
    
    print('All inline tests passed.')

if __name__ == '__main__':
    _run_tests()
