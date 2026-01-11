from typing import Counter
import numpy as np
import cupy as cp
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from typing import Optional


# =========================================================
#   Criteria
# =========================================================
def gini(counts):
    total = np.sum(counts)
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)

def entropy(counts):
    total = np.sum(counts)
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

# =========================================================
#   Tree Node
# =========================================================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # class prediction for leaf


# =========================================================
#   Decision Tree Classifier (Optimized)
# =========================================================
class DecisionTree:
    def __init__(self, criterion="gini", max_depth=20, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
        if criterion == "gini":
            self.criterion_func = gini
        elif criterion == "entropy":
            self.criterion_func = entropy
        else:
            raise ValueError("criterion must be 'gini' or 'entropy'")

    # ----------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)

    # ----------------------------------------------------
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature, best_threshold = None, None
        parent_counts = np.bincount(y, minlength=self.n_classes)
        parent_impurity = self.criterion_func(parent_counts)

        for feature in range(n_features):
            # Sort values for efficient threshold search
            sorted_idx = X[:, feature].argsort()
            x_sorted = X[sorted_idx, feature]
            y_sorted = y[sorted_idx]

            left_counts = np.zeros(self.n_classes)
            right_counts = parent_counts.copy()

            # Loop only over unique splits
            for i in range(1, n_samples):
                c = y_sorted[i - 1]
                left_counts[c] += 1
                right_counts[c] -= 1

                if x_sorted[i] == x_sorted[i - 1]:
                    continue  # skip identical values

                if i < self.min_samples_leaf or (n_samples - i) < self.min_samples_leaf:
                    continue

                left_imp = self.criterion_func(left_counts)
                right_imp = self.criterion_func(right_counts)
                p_left = i / n_samples

                gain = parent_impurity - (p_left * left_imp + (1 - p_left) * right_imp)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (x_sorted[i] + x_sorted[i - 1]) / 2

        return best_feature, best_threshold, best_gain

    # ----------------------------------------------------
    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        counts = np.bincount(y, minlength=self.n_classes)
        prediction = np.argmax(counts)

        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return Node(value=prediction)

        # Compute best split
        feature, threshold, gain = self._best_split(X, y)

        if gain == -1:
            return Node(value=prediction)

        # Split data
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)

    # ----------------------------------------------------
    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(row, self.root) for row in X])
    

EPS = 1e-8

def df_to_gpu_array(X: pd.DataFrame, dtype=cp.float32):
    """
    Convert a pandas DataFrame to a CuPy 2D array (float32) suitable for GPU KNN.
    Attempts:
      - bool -> 0/1
      - numeric left as-is
      - object / category -> categorical codes (int)
      - if conversion fails for a column, it will try get_dummies()
    Returns: cp.ndarray (n_samples, n_features)
    """
    X2 = X.copy()
    # booleans -> int
    bool_cols = X2.select_dtypes(include=["bool"]).columns
    for c in bool_cols:
        X2[c] = X2[c].astype(int)

    # object columns: try to convert to numeric, else to categorical codes
    obj_cols = X2.select_dtypes(include=["object", "category"]).columns
    for c in obj_cols:
        # try numeric conversion
        converted = pd.to_numeric(X2[c], errors="coerce")
        if converted.notna().all():
            X2[c] = converted
        else:
            # convert to categorical codes (unknown -> -1)
            X2[c] = pd.Categorical(X2[c]).codes.astype("int32")

    # final attempt: fill NaN with 0 (user should ideally handle missingness)
    X2 = X2.fillna(0)
    arr = X2.values.astype(np.float32, copy=False)
    return cp.asarray(arr, dtype=dtype)


class KNN_GPU:
    def __init__(self, k: int = 5, task: str = "classification"):
        """
        task: 'classification' or 'regression'
        """
        assert task in ("classification", "regression")
        self.k = int(k)
        self.task = task
        self.X_train_gpu = None
        self.y_train_gpu = None
        self.y_dtype = None

    def fit(self, X_train_gpu: cp.ndarray, y_train_gpu: cp.ndarray):
        """
        X_train_gpu: cp.ndarray shape (n_train, n_features), float32
        y_train_gpu: cp.ndarray shape (n_train,) - integers for classification, floats for regression
        """
        # Basic checks
        if not isinstance(X_train_gpu, cp.ndarray):
            raise TypeError("X_train_gpu must be a cupy ndarray. Use df_to_gpu_array or cp.asarray.")
        if not isinstance(y_train_gpu, cp.ndarray):
            raise TypeError("y_train_gpu must be a cupy ndarray.")
        if X_train_gpu.ndim != 2:
            raise ValueError("X_train_gpu must be 2D")
        if y_train_gpu.ndim != 1:
            raise ValueError("y_train_gpu must be 1D")

        self.X_train_gpu = X_train_gpu.astype(cp.float32, copy=False)
        self.y_train_gpu = y_train_gpu
        self.y_dtype = y_train_gpu.dtype

        # precompute norms of train rows
        self._train_sq = cp.sum(self.X_train_gpu * self.X_train_gpu, axis=1)  # shape (n_train,)

    def _predict_chunk(self, Xq: cp.ndarray, weighted: bool):
        """
        Predict for chunk Xq (cp.ndarray shape (m, d))
        Returns cp.ndarray shape (m,)
        """
        # Xq float32
        Xq = Xq.astype(cp.float32, copy=False)
        m = Xq.shape[0]

        # compute squared distances using (a-b)^2 = a^2 + b^2 - 2ab
        # shapes: Xq_sq (m,1), train_sq (1,n), cross (m,n)
        Xq_sq = cp.sum(Xq * Xq, axis=1)[:, None]            # (m,1)
        cross = Xq.dot(self.X_train_gpu.T)                  # (m,n)
        # distances squared (no negative due to numerical errors)
        d2 = Xq_sq - 2.0 * cross + self._train_sq[None, :]  # (m,n)
        # numeric safety
        cp.maximum(d2, 0, out=d2)

        # we don't necessarily need the sqrt, but for weighted inverse-distance use sqrt
        # get k smallest indices with argpartition (fast)
        n_train = self.X_train_gpu.shape[0]
        k = min(self.k, n_train)
        # indices of k smallest (unsorted)
        idx_k = cp.argpartition(d2, kth=k-1, axis=1)[:, :k]  # (m, k)

        # for stable ordering (optional): sort those k by distance
        rows = cp.arange(m)[:, None]
        k_dists = d2[rows, idx_k]
        order = cp.argsort(k_dists, axis=1)
        idx_k_sorted = idx_k[rows, order]           # (m,k)
        k_dists_sorted = cp.sqrt(k_dists[rows, order] + EPS)  # (m,k), add EPS to avoid zero

        if self.task == "regression":
            if weighted:
                weights = 1.0 / (k_dists_sorted + EPS)
                numer = cp.sum(weights * self.y_train_gpu[idx_k_sorted], axis=1)
                denom = cp.sum(weights, axis=1)
                preds = numer / (denom + EPS)
            else:
                preds = cp.mean(self.y_train_gpu[idx_k_sorted], axis=1)
            return preds

        # classification
        # if integer labels use bincount, else do unique+counts fallback
        y_neighbors = self.y_train_gpu[idx_k_sorted]  # (m,k)
        # If weighted voting:
        if weighted:
            weights = 1.0 / (k_dists_sorted + EPS)     # closer -> bigger weight
            # accumulate weighted votes per label for each sample
            # approach: for small number of classes, use bincount with weights; we'll try generic method:
            preds = cp.empty((m,), dtype=self.y_dtype)
            # vectorized but loop over batch rows (k usually small so this is fine)
            for i in range(m):
                labels, inv = cp.unique(y_neighbors[i], return_inverse=True)
                w = cp.zeros(labels.shape, dtype=cp.float32)
                cp.scatter_add(w, inv, weights[i])
                # choose label with max weight; tie -> smallest label due to argmax behaviour
                arg = int(cp.argmax(w))
                preds[i] = labels[arg]
            return preds

        # unweighted majority vote: use bincount per row (fast if labels are non-negative ints and small)
        if cp.issubdtype(self.y_dtype, cp.integer) and cp.min(self.y_train_gpu) >= 0:
            # find max label to size bincount
            max_label = int(cp.max(self.y_train_gpu))
            # to avoid huge bincount arrays if labels sparse, we fallback if too large
            if max_label <= 10000:
                preds = cp.empty((m,), dtype=self.y_dtype)
                # compute counts per row
                for i in range(m):
                    counts = cp.bincount(y_neighbors[i].astype(cp.int32), minlength=max_label+1)
                    preds[i] = int(cp.argmax(counts))
                return preds

        # generic fallback: count with unique
        preds = cp.empty((m,), dtype=self.y_dtype)
        for i in range(m):
            labels, counts = cp.unique(y_neighbors[i], return_counts=True)
            preds[i] = labels[int(cp.argmax(counts))]
        return preds

    def predict(self, Xq_gpu: cp.ndarray, batch_size: int = 4096, weighted: bool = False):
        """
        Predict labels for Xq_gpu (cp.ndarray) in batches.
        batch_size: number of query samples per chunk (tune based on GPU memory)
        weighted: if True, use inverse-distance weighting
        Returns cp.ndarray of predictions (on GPU)
        """
        if self.X_train_gpu is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if not isinstance(Xq_gpu, cp.ndarray):
            raise TypeError("Xq_gpu must be a cupy ndarray. Use df_to_gpu_array or cp.asarray.")

        n = Xq_gpu.shape[0]
        preds = cp.empty((n,), dtype=self.y_dtype if self.task == "classification" else cp.float32)

        start = 0
        while start < n:
            stop = min(n, start + batch_size)
            Xbatch = Xq_gpu[start:stop]
            try:
                preds_batch = self._predict_chunk(Xbatch, weighted=weighted)
            except cp.cuda.memory.OutOfMemoryError:
                # if OOM, reduce batch size and retry
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"[KNN_GPU] OOM for batch_size; retrying with batch_size={batch_size}")
                continue
            preds[start:stop] = preds_batch
            start = stop

        return preds
    

class RandomForestScratch:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Création de l'arbre
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            
            # Bootstrapping (Échantillonnage avec remise)
            n_samples = X.shape[0]
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            print(f"Arbre {_ + 1}/{self.n_trees} entraîné.")

    def predict(self, X):
        # Récupérer les prédictions de tous les arbres
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Vote majoritaire
        # tree_preds shape: [n_trees, n_samples] -> on veut [n_samples]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        predictions = []
        for preds in tree_preds:
            most_common = Counter(preds).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
    