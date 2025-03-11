import numpy as np
from config import LINGUISTIC_SCALE

def reciprocal_fuzzy(fuzzy_number):
    l, m, u = fuzzy_number
    return (1/u, 1/m, 1/l)

def calculate_fahp(user_weights, criteria):
    debug_info = []
    debug_info.append("[FAHP DEBUG] Starting Fuzzy AHP Calculation...\n")

    n = len(criteria)
    pairwise_matrix = np.empty((n, n, 3), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                pairwise_matrix[i, j] = (1, 1, 1)
            elif i < j:
                key = user_weights.get(criteria[i], "equally_important")
                triplet = LINGUISTIC_SCALE[key]
                pairwise_matrix[i, j] = triplet
            else:
                pairwise_matrix[i, j] = reciprocal_fuzzy(pairwise_matrix[j, i])

    debug_info.append(f"[FAHP DEBUG] Pairwise Matrix:\n{pairwise_matrix}\n")

    # ✅ Move these inside the function so `pairwise_matrix` is accessible
    total_l = np.sum(pairwise_matrix[:, :, 0])  # Sum of all L (lower bound)
    total_m = np.sum(pairwise_matrix[:, :, 1])  # Sum of all M (middle value)
    total_u = np.sum(pairwise_matrix[:, :, 2])  # Sum of all U (upper bound)

    synthetic_extents = [
        (
            np.sum(pairwise_matrix[i, :, 0]) / total_u,  # L component
            np.sum(pairwise_matrix[i, :, 1]) / total_m,  # M component
            np.sum(pairwise_matrix[i, :, 2]) / total_l   # U component
        )
        for i in range(n)
    ]

    debug_info.append("[FAHP DEBUG] Synthetic Extents (L, M, U):\n" + str(synthetic_extents) + "\n")

    # Compute final normalized FAHP weights using the middle value
    normalized_weights = np.array([extent[1] for extent in synthetic_extents]) / np.sum([extent[1] for extent in synthetic_extents])

    debug_info.append("[FAHP DEBUG] Normalized Weights (Updated): " + str(normalized_weights) + "\n")

    return normalized_weights, "\n".join(debug_info)
