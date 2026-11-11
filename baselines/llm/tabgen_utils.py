import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from scipy import stats
from scipy.stats import wasserstein_distance_nd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from geomloss import SamplesLoss
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)

import pdb

def wasserstein_distance(X: np.ndarray, X_syn: np.ndarray) -> float:
    """
    Compute Wasserstein distance between original data and synthetic data.

    Args:
        X: Original data (numpy array)
        X_syn: Synthetically generated data (numpy array)

    Returns:
        WD_value: Wasserstein distance (float)
    """
    # Reshape data if necessary
    X_ = X.reshape(len(X), -1)
    X_syn_ = X_syn.reshape(len(X_syn), -1)

    # Ensure both datasets have the same number of samples
    if len(X_) > len(X_syn_):
        X_syn_ = np.concatenate([X_syn_, np.zeros((len(X_) - len(X_syn_), X_.shape[1]))])

    # Normalize data
    scaler = MinMaxScaler().fit(X_)
    X_ = scaler.transform(X_)
    X_syn_ = scaler.transform(X_syn_)

    # Convert to tensors
    X_ten = torch.from_numpy(X_)
    Xsyn_ten = torch.from_numpy(X_syn_)

    # Compute Wasserstein distance using Sinkhorn algorithm
    OT_solver = SamplesLoss(loss="sinkhorn")
    WD_value = OT_solver(X_ten, Xsyn_ten).cpu().numpy().item()

    return WD_value


def jensen_shannon_distance(X_gt: np.ndarray, X_syn: np.ndarray, normalize: bool = True, n_bins: int = 10, laplace_smoothing: bool = True) -> float:
    """
    Evaluate the average Jensen-Shannon distance between two 2D numpy arrays.

    Args:
        X_gt (np.ndarray): Ground truth data (2D array).
        X_syn (np.ndarray): Synthetic data (2D array).
        normalize (bool): If True, normalize the value counts to probabilities.
        n_bins (int): Maximum number of bins for histogram.
        laplace_smoothing (bool): If True, apply Laplace smoothing to avoid zero probabilities.

    Returns:
        float: The average Jensen-Shannon distance across all columns.
    """
    if X_gt.shape[1] != X_syn.shape[1]:
        raise ValueError("The number of columns in X_gt and X_syn must be the same.")
    
    jsd_values = []

    # Iterate over each column
    for i in range(X_gt.shape[1]):
        try:
            # Get the column data
            col_gt = X_gt[:, i]
            col_syn = X_syn[:, i]
            
            if col_gt.size == 0 or col_syn.size == 0:
                continue
            
            # Determine the number of unique values in the ground truth column for binning
            local_bins = min(n_bins, len(np.unique(col_gt)))
            
            # Create histograms for both ground truth and synthetic data
            gt_hist, gt_bins = np.histogram(col_gt, bins=local_bins, density=normalize)
            syn_hist, _ = np.histogram(col_syn, bins=gt_bins, density=normalize)
            
            # Apply Laplace smoothing if needed
            if laplace_smoothing:
                gt_hist += 1
                syn_hist += 1
            
            # Normalize histograms again after smoothing
            gt_hist = gt_hist / np.sum(gt_hist)
            syn_hist = syn_hist / np.sum(syn_hist)

            # Compute Jensen-Shannon distance for the column
            jsd_value = distance.jensenshannon(gt_hist, syn_hist)
            if np.isnan(jsd_value):
                raise RuntimeError(f"NaNs in JSD calculation for column {i}")
            
            jsd_values.append(jsd_value)
        except:
            pass            

    # Return the average JSD across all columns
    return np.mean(jsd_values)

def compute_distance(df, sampled_indices_collection, df_tmp, over_sampling=True, dist_type=None, prompt_idx=0):
    assert dist_type is None

    distances = []

    for sampled_idx in sampled_indices_collection:
        sampled_df = df.iloc[sampled_idx]
        combined_df = pd.concat([sampled_df, df_tmp], axis=0)

        encoded_gt = []
        encoded_syn = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                #print(col, 'cat')
                cat_col_gt = df[col].to_numpy().reshape(-1, 1)
                cat_col_syn = combined_df[col].to_numpy().reshape(-1, 1)

                enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_cat_col_gt = enc.fit_transform(cat_col_gt)
                encoded_cat_col_syn = enc.transform(cat_col_syn)

                encoded_gt.append(encoded_cat_col_gt)
                encoded_syn.append(encoded_cat_col_syn)
            else: 
                #print(col, 'num')
                encoded_gt.append(df[col].to_numpy().reshape(-1, 1))
                encoded_syn.append(combined_df[col].to_numpy().reshape(-1, 1))
        
        combined_encoded_df = np.concatenate(encoded_gt, axis=1) #gt
        combined_encoded = np.concatenate(encoded_syn, axis=1) #syn

        if over_sampling:
            combined_encoded = resample(combined_encoded, replace=True, n_samples=combined_encoded_df.shape[0], random_state=2024)
        
        if dist_type == 'jensenshannon' or prompt_idx % 2 == 0:
            dist = jensen_shannon_distance(combined_encoded, combined_encoded_df)
        
        elif dist_type == 'kstest' or prompt_idx % 2 == 1:
            dist = 0
            for col_idx in range(combined_encoded.shape[1]):
                cur_dist = stats.kstest(combined_encoded[:, col_idx], combined_encoded_df[:, col_idx]).statistic_location
                dist += cur_dist
            dist = dist / combined_encoded.shape[1]
        
        if dist_type == 'wasserstein' or prompt_idx % 10 == 0:
            print('wasserstein')
            dist = wasserstein_distance(combined_encoded, combined_encoded_df)
        
        distances.append(dist)

    return distances

def get_indices(df, col, dtype, max_num=500, num=100):
    subset_collection = []
    if dtype == 'int64' or dtype == 'float64':
        quantize_col = np.linspace(df[col].min(), df[col].max(), num=num)
        for l,r in zip(quantize_col[:-1], quantize_col[1:]):
            subset_idx = df[(df[col] >= l) & (df[col] < r)].index
            if len(subset_idx) > max_num:
                subset_idx = np.random.choice(subset_idx, size=max_num, replace=False)
            subset_collection.append(subset_idx)
        return subset_collection
    elif dtype == 'object':
        for cls in df[col].unique():
            subset_idx = df[df[col] == cls].index
            if len(subset_idx) > max_num:
                subset_idx = np.random.choice(subset_idx, size=max_num, replace=False)
            subset_collection.append(subset_idx)
        return subset_collection
    else:
        raise ValueError('Unsupported dtype: ' + dtype)
