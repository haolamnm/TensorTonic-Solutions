# def contains(inner: list[int], outer: list[int]) -> bool:
#     k = 0
#     cnt = 0
#     for i in range(len(inner)):
#         # print(f"inner[{i}]", i, inner[i])
#         for j in range(k, len(outer)):
#             # print(f"outer[{j}]", j, outer[j])
#             if inner[i] == outer[j]:
#                 k = j + 1
#                 cnt += 1
#                 break
#     # print(cnt, len(inner))
#     return cnt == len(inner)

def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # hitrate@k is just "did any relevant item appear in the top-k? order doesn't matter, one hit is enough."

    scores = [
        bool(set(g) & set(r[:k]))
        for r, g in zip(recommendations, ground_truth)
    ]
    return sum(scores) / len(scores)
    
    

    