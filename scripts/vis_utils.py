import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

def visualize_results(
    data_filepath,
    rental_filepath,
    zip_filepath,
    time_filepath,
    features_filepath,
    property_types_filepath,
    results,
    number_of_samples=10,
):
    """Visualize predicted vs. historical housing and rental prices.
    
    Args:
        data_filepath (str): Path to the housing data file (numpy array with shape [zipcodes x time x features x property_type]).
        rental_filepath (str): Path to the rental data file (numpy array with shape [zipcodes x time]).
        zip_filepath (str): Path to the file containing zipcode data.
        time_filepath (str): Path to the file containing the time array.
        features_filepath (str): Path to the file containing feature names.
        property_types_filepath (str): Path to the file containing property type names.
        results (dict): Dictionary with keys:
            - "housing_pred": Housing predictions (numpy array with shape [num_samples, forecast_length, num_property_types]).
            - "rental_pred": Rental predictions (numpy array with shape [num_samples, forecast_length, 1] or [num_samples, forecast_length]).
            - "zipcodes": Zipcodes corresponding to the predictions.
            - "time": Forecast time points.
        number_of_samples (int, optional): Number of random sample zipcodes to visualize. Default is 10.
    
    Returns:
        None
    """
    data = np.load(data_filepath)
    rental_data = np.load(rental_filepath)
    zipcodes = np.load(zip_filepath)
    time_arr = np.load(time_filepath, allow_pickle=True)
    features = np.load(features_filepath)
    property_types = np.load(property_types_filepath)

    housing_pred = results["housing_pred"]
    rental_pred = results["rental_pred"]
    pred_zipcodes = results["zipcodes"]
    forecast_times = results["time"]

    feature_idx = np.where(features == "median_sale_price")[0][0]
    sample_zips = random.sample(list(np.unique(pred_zipcodes)), number_of_samples)
    sample_mask = np.isin(zipcodes, sample_zips)
    sample_indices = np.nonzero(sample_mask)[0]
    historical_times = time_arr[-36:]
    
    n_housing = len(property_types)
    total_rows = n_housing + 1

    fig = plt.figure(figsize=(16, 4 * total_rows))
    gs = gridspec.GridSpec(nrows=total_rows, ncols=1)
    
    for p in range(total_rows):
        ax = fig.add_subplot(gs[p, 0])
        if p < n_housing:
            hist = data[sample_indices, -36:, feature_idx, p]
            fore = housing_pred[sample_indices, :, p]
            ft = np.tile(forecast_times, (hist.shape[0], 1))
            full_x = np.hstack([np.tile(historical_times, (hist.shape[0], 1)), ft])
            full_y = np.hstack([hist, fore])
            for i in range(full_x.shape[0]):
                ax.plot(full_x[i], full_y[i], marker='o', ms=2, linewidth=0.5)
            ax.axvline(x=historical_times[-1], color='gray', linestyle='--', linewidth=0.5)
            ax.set_title(property_types[p])
            ax.set_xlabel("Time")
            ax.set_yscale('log')
            if p == 0:
                ax.set_ylabel("Median Sale Price")
        else:
            if rental_pred.ndim == 3:
                rental_pred = rental_pred.squeeze(-1)
            hist_rental = rental_data[sample_indices, -36:]
            fore_rental = rental_pred[sample_indices, :]
            ft_rental = np.tile(forecast_times, (hist_rental.shape[0], 1))
            full_x_rental = np.hstack([np.tile(historical_times, (hist_rental.shape[0], 1)), ft_rental])
            full_y_rental = np.hstack([hist_rental, fore_rental])
            for i in range(full_x_rental.shape[0]):
                ax.plot(full_x_rental[i], full_y_rental[i], marker='o', ms=2, linewidth=0.5)
            ax.axvline(x=historical_times[-1], color='gray', linestyle='--', linewidth=0.5)
            ax.set_title("Rental")
            ax.set_xlabel("Time")
            ax.set_ylabel("Rental Price")
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    

def ensure_all_csvs(results_filepath, data_filepath, rental_filepath):
    """Converts all .npy files needed for visualization to .csv and stores them in the visualization/static folder.
    
    Args:
        results_filepath (str): Path to the results data file (dictionary of predictions)
        data_filepath (str): Path to the housing data file (numpy array with shape [zipcodes x time x features x property_type])
        rental_filepath (str): Path to the rental data file (numpy array with shape [zipcodes x time]).
    
    Returns:
        None
    """
    # --- Part 1: rental_pred CSV ---
    results = np.load(results_filepath, allow_pickle=True).item()
    zip_arr = np.array(results['zipcodes']).reshape(-1, 1)

    
    rental_pred = results["rental_pred"]
    if rental_pred.ndim == 3 and rental_pred.shape[2] == 1:
        rental_pred = rental_pred.squeeze(-1)

    # build header for full rental preds
    header = ['Zip code'] + [str(i) for i in range(1, rental_pred.shape[1] + 1)]
    rental_data = np.hstack((zip_arr, rental_pred))
    df_rental_full = pd.DataFrame(rental_data, columns=header)
    df_rental_full.to_csv("visualization/static/Rental_pred.csv", index=False)

    # --- Part 1b: rental.csv with Zip code + rounded most recent rental column ---
    # extract most recent rental column
    rental_first = np.load(rental_filepath)[:, -1]
    # round and cast to int
    rental_first_rounded = np.round(rental_first).astype(int)
    # cast zip codes to int as well
    zip_int = zip_arr.flatten().astype(int)

    df_rental_first2 = pd.DataFrame({
        'Zip code': zip_int,
        'rental': rental_first_rounded
    })
    df_rental_first2.to_csv("visualization/static/rental.csv", index=False)

    # --- Part 2: housing_pred CSVs with custom names ---
    labels = [
        'All Residential',
        'Condo',
        'Multi_Family',
        'Single_Family',
        'Townhouse'
    ]
    y_pred = results['housing_pred']  # shape: (n, 12, 5)

    for idx, label in enumerate(labels):
        mat = y_pred[:, :, idx]       # (n,12)
        data = np.hstack((zip_arr, mat))
        df = pd.DataFrame(data, columns=header)

        safe_name = label.replace("/", "-").replace(" ", "_")
        df.to_csv(f"visualization/static/{safe_name}.csv", index=False)

    for label in labels:
        safe_name = label.replace("/", "-").replace(" ", "_")

    # --- Part 3: median_sale_price CSV with rounding and zip codes ---
    data2 = np.load(data_filepath, mmap_mode="r+")
    median_sale_price = data2[:, -1, 7, 0]
    median_sale_price = np.round(median_sale_price).astype(int)

    df_median = pd.DataFrame({
        'Zip code': zip_int,
        'median_sale_price': median_sale_price
    })
    df_median.to_csv("visualization/static/median_sale_price.csv", index=False)
