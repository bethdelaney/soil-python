import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance

# Utility function to compute Mahalanobis distance
def mahalanobis_distance(x, mean, inv_cov):
    return distance.mahalanobis(x, mean, inv_cov)

# Main function to implement the Puchwein algorithm
def puchwein(X, pc=0.95, k=0.2, min_sel=5, details=False, center=True, scale=False):
    # Ensure X is a 2D numpy array
    X = np.array(X)
    
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 columns")
    if min_sel >= X.shape[0]:
        raise ValueError("min_sel must be lower than the number of rows in X")
    
    # Step 1: PCA with centering and scaling
    pca = PCA(n_components=X.shape[1])
    X_transformed = pca.fit_transform(X)
    
    # Determine the number of principal components based on explained variance
    if pc < 1:
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(explained_variance >= pc) + 1
    else:
        num_components = int(pc)
    
    X_reduced = X_transformed[:, :num_components]
    
    # Scale the principal components
    X_scaled = X_reduced / np.std(X_reduced, axis=0)
    
    # Step 2: Compute Mahalanobis distance to the center
    center = np.mean(X_scaled, axis=0)
    cov_matrix = np.cov(X_scaled, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    mahal_distances = np.array([mahalanobis_distance(sample, center, inv_cov_matrix) for sample in X_scaled])
    
    # Sort by Mahalanobis distance
    order = np.argsort(mahal_distances)[::-1]
    X_scaled = X_scaled[order]
    mahal_distances = mahal_distances[order]
    
    # Initial limiting distance
    d_ini = k * max((num_components - 2), 1)
    
    selected_indices = []
    all_selected_loops = []
    m = 1
    
    while True:
        Dm = m * d_ini
        selected = [0]  # Always start with the first sample (largest distance)
        
        for i in range(1, len(X_scaled)):
            dist_to_selected = np.min([distance.euclidean(X_scaled[i], X_scaled[j]) for j in selected])
            if dist_to_selected > Dm:
                selected.append(i)
        
        all_selected_loops.append(selected)
        
        if len(selected) <= min_sel:
            break
        m += 1
    
    # Step 3: Compute leverage and optimize
    X_matrix = X_scaled
    leverage = np.diag(X_matrix @ np.linalg.pinv(X_matrix.T @ X_matrix) @ X_matrix.T)
    observed_leverage = [np.sum(leverage[selected]) for selected in all_selected_loops]
    theoretical_leverage = (np.sum(leverage) / len(leverage)) * np.array([len(selected) for selected in all_selected_loops])
    leverage_diff = np.array(observed_leverage) - theoretical_leverage
    
    # Find the optimal loop with the maximum leverage difference
    optimal_loop = np.argmax(leverage_diff)
    model_indices = all_selected_loops[optimal_loop]
    
    if details:
        return {
            'model': order[model_indices],
            'test': np.setdiff1d(np.arange(len(X)), order[model_indices]),
            'pc': X_scaled,
            'loop_optimal': optimal_loop,
            'leverage': {
                'loop': np.arange(1, m),
                'observed_leverage': observed_leverage,
                'theoretical_leverage': theoretical_leverage,
                'leverage_diff': leverage_diff
            },
            'details': all_selected_loops
        }
    else:
        return {
            'model': order[model_indices],
            'test': np.setdiff1d(np.arange(len(X)), order[model_indices]),
            'pc': X_scaled,
            'loop_optimal': optimal_loop,
            'leverage': {
                'loop': np.arange(1, m),
                'observed_leverage': observed_leverage,
                'theoretical_leverage': theoretical_leverage,
                'leverage_diff': leverage_diff
            }
        }
