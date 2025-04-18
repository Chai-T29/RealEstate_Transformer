import os
import requests
import re
import gzip
import glob
import shutil
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d, splrep, BSpline
from tqdm.notebook import tqdm


import os
import requests
import gzip
import pandas as pd
import numpy as np

def process_redfin_data(url, save_path, max_na_threshold=0.1, period='period_begin', overwrite=False, chunk_size=1024):
    """Download, extract, clean, and reshape Redfin data.
    
    Args:
        url (str): URL to download file.
        save_path (str): Directory to save processed files.
        max_na_threshold (float): Maximum NA fraction allowed per column.
        period (str): 'period_begin' or 'period_end'.
        overwrite (bool): If True, overwrite existing processed files.
        chunk_size (int): Chunk size for downloading.
    
    Returns:
        None
    """
    suffix = '_pb' if period == 'period_begin' else '_pe'
    data_filepath = os.path.join(save_path, f"clean_redfin_data{suffix}.npy")
    zip_filepath = os.path.join(save_path, f"clean_redfin_zipcodes{suffix}.npy")
    time_filepath = os.path.join(save_path, f"clean_redfin_timestamps{suffix}.npy")
    features_filepath = os.path.join(save_path, f"clean_redfin_features{suffix}.npy")
    property_types_filepath = os.path.join(save_path, f"clean_redfin_property_types{suffix}.npy")

    if all(os.path.exists(p) for p in [data_filepath, zip_filepath, time_filepath, features_filepath, property_types_filepath]) and not overwrite:
        print("Cleaned data already exists. Skipping download and processing.")
        return

    temp_download_path = os.path.join(save_path, "temp_redfin_download.gz")

    print(f"Downloading file from {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(temp_download_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
        print(f"File downloaded successfully: {temp_download_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return

    print("Starting data cleaning process")
    if period not in ['period_begin', 'period_end']:
        raise ValueError("The period parameter must be either 'period_begin' or 'period_end'.")

    drop_patterns = [
        '_id', '_mom', '_yoy', '_updated', 'region_type',
        'region_type_id', 'table_id', 'is_seasonally_adjusted', 'city', 'state',
        'state_code', 'parent_metro_region', 'parent_metro_region_metro_code',
        'last_updated'
    ]
    drop_patterns.append('period_end' if period == 'period_begin' else 'period_begin')

    with gzip.open(temp_download_path, 'rt') as f:
        header_df = pd.read_csv(f, delimiter="\t", nrows=0)
    all_columns = header_df.columns.tolist()
    cols_to_load = [
        col for col in all_columns
        if (col == 'region' or col == 'property_type' or col == period or not any(col.endswith(pat) for pat in drop_patterns))
    ]
    print(f"Columns to load: {len(cols_to_load)} selected from {len(all_columns)} total")
    
    with gzip.open(temp_download_path, 'rt') as f:
        df = pd.read_csv(f, delimiter="\t", usecols=cols_to_load)
    print(f"Raw loaded data shape: {df.shape}")

    df['zip_code'] = df['region'].astype(str).apply(lambda x: x.split(':')[-1].strip())
    df.drop(columns='region', inplace=True)
    df = df[df['zip_code'].str.match(r'^\d{5}$', na=False)]
    df[period] = pd.to_datetime(df[period])
    df['property_type'] = df['property_type'].astype(str)

    valid_cols = df.columns[df.isna().mean() <= max_na_threshold]
    df = df[valid_cols]
    print(f"Remaining columns after NA filter: {df.shape[1]}")

    pivoted = df.pivot_table(
        index=['zip_code', 'property_type', period],
        values=[c for c in df.columns if c not in ['zip_code', 'property_type', period]],
        aggfunc='max'
    )

    features = list(pivoted.columns)
    zip_codes = sorted(pivoted.index.get_level_values('zip_code').unique())
    property_types = sorted(pivoted.index.get_level_values('property_type').unique())
    timestamps = sorted(pivoted.index.get_level_values(period).unique())

    print(f"Number of zip codes: {len(zip_codes)}")
    print(f"Number of property types: {len(property_types)}")
    print(f"Number of timestamps: {len(timestamps)}")
    print(f"Number of features: {len(features)}")

    full_index = pd.MultiIndex.from_product([zip_codes, property_types, timestamps],
                                              names=['zip_code', 'property_type', period])
    df_complete = pivoted.reindex(full_index)
    df_complete = df_complete.reorder_levels(['zip_code', period, 'property_type'])

    n_zip = len(zip_codes)
    n_time = len(timestamps)
    n_feat = len(features)
    n_ptype = len(property_types)

    data_array = np.full((n_zip, n_time, n_feat, n_ptype), np.nan)
    for f_idx, feature in enumerate(features):
        feature_vals = df_complete[feature].values
        feature_reshaped = feature_vals.reshape(n_zip, n_time, n_ptype)
        data_array[:, :, f_idx, :] = feature_reshaped

    np.save(data_filepath, data_array)
    np.save(zip_filepath, np.array(zip_codes))
    np.save(time_filepath, np.array(timestamps))
    np.save(features_filepath, np.array(features))
    np.save(property_types_filepath, np.array(property_types))

    print("Data cleaning complete and files saved.")

    if os.path.exists(temp_download_path):
        os.remove(temp_download_path)
        print(f"Removed file: {temp_download_path}")


def process_rental_data(rental_csv_dir, zip_filepath, time_filepath):
    """Combined processing of rental data.

    Args:
        rental_csv_dir (str): Path to the directory containing raw rental CSV files.
        zip_filepath (str): Path to the master ZIP codes NumPy file.
        time_filepath (str): Path to the timestamps NumPy file.

    Returns:
        str: The file path of the final processed rental data.
    """
    for filename in os.listdir(rental_csv_dir):
        if not filename.endswith('-Data.csv'):
            file_path = os.path.join(rental_csv_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")

    save_path = zip_filepath.split("/clean_redfin")[0] + "/clean_rental_data.npy"
    if os.path.exists(save_path):
        print(f"File '{save_path}' already exists. Skipping processing.")
        return save_path

    csv_files = glob.glob(os.path.join(rental_csv_dir, '*-Data.csv'))
    if not csv_files:
        print(f"No CSV files ending with '-Data.csv' found in {rental_csv_dir}")
        return

    dfs = []
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        match = re.search(r'(\d{4})', filename)
        if match:
            year = int(match.group(1))
        else:
            print(f"Year not found in filename {filename}, skipping this file.")
            continue

        try:
            df_temp = pd.read_csv(filepath, header=1)
            df_temp['Year'] = year
            dfs.append(df_temp)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not dfs:
        print("No valid CSV data found to process.")
        return

    df = pd.concat(dfs, ignore_index=True)
    try:
        df = df[['Geographic Area Name', 'Estimate!!Median gross rent', 'Year']]
    except KeyError as e:
        print("Expected columns are missing from the input data:", e)
        return

    df.rename(columns={
        'Geographic Area Name': 'Zip',
        'Estimate!!Median gross rent': 'Median_Gross_Rent'
    }, inplace=True)
    df['Zip'] = df['Zip'].astype(str).str[-5:]
    df = df[pd.to_numeric(df['Median_Gross_Rent'], errors='coerce').notna()]
    df['Zip'] = df['Zip'].astype(str).str.zfill(5)

    try:
        master_zip_array = np.load(zip_filepath)
    except Exception as e:
        print("Error loading master ZIP code file:", e)
        return
    master_zip_df = pd.DataFrame(master_zip_array.astype(str), columns=['Zip'])
    master_zip_df['Zip'] = master_zip_df['Zip'].str.strip().str.zfill(5)

    df['Zip'] = df['Zip'].astype(str).str.strip().str.zfill(5)
    df['Year'] = df['Year'].astype(int)
    df['Median_Gross_Rent'] = pd.to_numeric(df['Median_Gross_Rent'], errors='coerce')

    years = list(range(2013, 2024))
    all_combinations = pd.MultiIndex.from_product(
        [master_zip_df['Zip'], years],
        names=['Zip', 'Year']
    ).to_frame(index=False)

    merged = all_combinations.merge(
        df[['Zip', 'Year', 'Median_Gross_Rent']],
        on=['Zip', 'Year'],
        how='left'
    )

    master_zip_df['zip_order'] = range(len(master_zip_df))
    merged = merged.merge(master_zip_df[['Zip', 'zip_order']], on='Zip', how='left')
    merged = merged.sort_values(by=['zip_order', 'Year']).drop(columns='zip_order').reset_index(drop=True)

    pivot_df = merged.pivot(index='Zip', columns='Year', values='Median_Gross_Rent')
    pivot_df = pivot_df.reset_index()  # Ensure 'Zip' becomes a column.
    pivot_array = pivot_df.to_numpy()
    print("Pivot rental data processed in memory.")

    raw_rental_data = pivot_array

    try:
        master_zipcodes = np.load(zip_filepath)
    except Exception as e:
        print("Error loading master ZIP code file:", e)
        return
    try:
        times = np.load(time_filepath, allow_pickle=True)
    except Exception as e:
        print("Error loading timestamp file:", e)
        return

    year_to_time_index = {}
    for idx, time_val in enumerate(times):
        t = pd.Timestamp(time_val)
        if t.month == 12:
            year_to_time_index[t.year] = idx

    master_data = np.full((len(master_zipcodes), len(times)), np.nan, dtype=np.float32)
    rental_dict = {}

    for row in raw_rental_data:
        zipcode = row[0]
        try:
            values = np.array(row[1:], dtype=np.float32)
        except ValueError:
            values = np.array([np.nan] * (len(row) - 1), dtype=np.float32)
        rental_dict[zipcode] = values

    for i, zipcode in enumerate(master_zipcodes):
        if zipcode in rental_dict:
            rental_values = rental_dict[zipcode]
            for j, year in enumerate(range(2013, 2024)):
                if year in year_to_time_index:
                    time_idx = year_to_time_index[year]
                    master_data[i, time_idx] = rental_values[j]

    np.save(save_path, master_data)
    print(f"Rental data saved to '{save_path}'.")
    return save_path


def interpolate_series(time, values, interp_method, smoothing_factor=None):
    """Interpolate and smooth a time series.
    
    Args:
        time (array-like): Time vector.
        values (array-like): Values to interpolate.
        interp_method (str): "linear", "spline", or "bspline".
        smoothing_factor (float, optional): Smoothing factor for spline.
    Returns:
        np.ndarray: Interpolated values.
    """
    time = np.array(time)
    values = np.array(values)
    valid_mask = (~np.isnan(values)) & (values != 0)
    valid_time = time[valid_mask]
    valid_values = values[valid_mask]

    if valid_time.size == 0:
        return np.zeros_like(time, dtype=float)
    elif valid_time.size == 1:
        base_val = valid_values.item()
        noise_scale = 0.01 * abs(base_val) if base_val != 0 else 0.01
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=time.shape)
        return np.full(time.shape, base_val, dtype=float) + noise

    interp_func = interp1d(valid_time, valid_values, bounds_error=False, fill_value="extrapolate")
    linear_values = interp_func(time)

    before_start = time < valid_time.min()
    if np.any(before_start):
        first_val = valid_values[0]
        noise_scale = 0.01 * abs(first_val) if first_val != 0 else 0.01
        linear_values[before_start] = first_val + np.random.normal(0.0, noise_scale, np.sum(before_start))

    after_end = time > valid_time.max()
    if np.any(after_end):
        last_val = valid_values[-1]
        noise_scale = 0.01 * abs(last_val) if last_val != 0 else 0.01
        linear_values[after_end] = last_val + np.random.normal(0.0, noise_scale, np.sum(after_end))

    if interp_method.lower() == "linear":
        return linear_values

    if smoothing_factor is None and interp_method.lower() == "spline":
        smoothing_factor = time.size * np.var(linear_values) * 0.1

    if interp_method.lower() == "spline":
        k = min(3, time.size - 1)
        try:
            spline = UnivariateSpline(time, linear_values, k=k, s=smoothing_factor)
            interp_values = spline(time)
        except Exception as e:
            print(f"UnivariateSpline failed with degree {k}: {e}. Falling back to linear interpolation.")
            interp_values = linear_values

    elif interp_method.lower() == "bspline":
        if smoothing_factor is None:
            smoothing_factor = 0
        k = min(3, time.size - 1)
        try:
            tck = splrep(time, linear_values, s=smoothing_factor, k=k)
            bspline = BSpline(*tck)
            interp_values = bspline(time)
        except Exception as e:
            print(f"BSpline interpolation failed with degree {k}: {e}. Falling back to linear interpolation.")
            interp_values = linear_values
    else:
        raise ValueError(f"Unknown interpolation method: {interp_method}")

    return interp_values


def impute_missing_values(data_filepath, rental_filepath, features_filepath, time_filepath, interp_method="none", overwrite=False):
    """Interpolate missing values in housing and rental datasets.
    
    Args:
        data_filepath (str): Path to housing data (4D np.ndarray).
        rental_filepath (str): Path to rental data (2D np.ndarray).
        features_filepath (str): Path to features.
        time_filepath (str): Path to timestamps.
        interp_method (str): "none", "linear", "spline", or "bspline".
        overwrite (bool): If True, overwrite existing imputed files.
    Returns:
        tuple: (imputed housing data filepath, imputed rental data filepath)
    """
    if interp_method == "none":
        return data_filepath, rental_filepath

    imputed_data_filepath = data_filepath.split("/clean_redfin")[0] + f"/{interp_method}_" + data_filepath.split("/")[-1]
    imputed_rental_filepath = rental_filepath.split("/clean_rental")[0] + f"/{interp_method}_" + rental_filepath.split("/")[-1]
    
    process = overwrite or not (os.path.exists(imputed_data_filepath) and os.path.exists(imputed_rental_filepath))
    
    if not process:
        print(f"Imputed files already exist at '{imputed_data_filepath}' and '{imputed_rental_filepath}'. Skipping processing.")
        return imputed_data_filepath, imputed_rental_filepath
    else:
        if overwrite:
            if os.path.exists(imputed_data_filepath):
                os.remove(imputed_data_filepath)
                print(f"Removing existing file at '{imputed_data_filepath}'")
            if os.path.exists(imputed_rental_filepath):
                os.remove(imputed_rental_filepath)
                print(f"Removing existing file at '{imputed_rental_filepath}'")
    
    data = np.load(data_filepath)
    rental_data = np.load(rental_filepath)
    features = np.load(features_filepath)
    times = np.load(time_filepath, allow_pickle=True)
    
    target_idx = np.where(features == "median_sale_price")[0][0]
    features_data = data[:, :, target_idx, :]
    features_data[features_data < 2000] = np.nan
    features_data[features_data > 5000000] = np.nan
    data[:, :, target_idx, :] = features_data

    num_zip, T, F, P = data.shape
    time_vector = np.arange(T)
    
    num_zip_rental, T_rental = rental_data.shape
    if num_zip != num_zip_rental or T != T_rental:
        raise ValueError("Housing data and rental data must have matching rows and time dimensions.")
    
    print("Starting imputation on both housing and rental data")
    for i in tqdm(range(num_zip), desc="Imputing data"):
        for f in range(F):
            for p in range(P):
                data[i, :, f, p] = interpolate_series(time_vector, data[i, :, f, p], interp_method=interp_method)
        rental_data[i, :] = interpolate_series(time_vector, rental_data[i, :], interp_method=interp_method)
    
    np.save(imputed_data_filepath, data)
    print(f"Housing imputed data saved at '{imputed_data_filepath}'")
    np.save(imputed_rental_filepath, rental_data)
    print(f"Rental imputed data saved at '{imputed_rental_filepath}'")
    
    return imputed_data_filepath, imputed_rental_filepath