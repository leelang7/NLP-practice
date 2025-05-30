import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise
import time

# 벡터 정의
sent_1 = np.array([0.3, 0.2, 0.2133, 0.3891, 0.8852, 0.586, 1.244, 0.777, 0.882] * 10000) 
sent_2 = np.array([0.03, 0.223, 0.1, 0.4, 2.931, 0.122, 0.5934, 0.8472, 0.54] * 10000)
sent_3 = np.array([0.13, 0.83, 0.827, 0.92, 0.1, 0.32, 0.28, 0.34, 0] * 10000)

all_sent = np.array([sent_1, sent_2, sent_3])

# 1️⃣ 사용자 정의 방식
def cosine_custom(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(v2, v2)))

start = time.perf_counter()
sim_custom = cosine_custom(sent_1, sent_2)
end = time.perf_counter()
print(f"[1. 사용자 정의] 유사도: {sim_custom:.6f} | 수행시간: {(end - start) * 1000:.6f} ms")

# 2️⃣ NumPy 전용 방식
def cosine_numpy(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

start = time.perf_counter()
sim_numpy = cosine_numpy(sent_1, sent_2)
end = time.perf_counter()
print(f"[2. NumPy-only ] 유사도: {sim_numpy:.6f} | 수행시간: {(end - start) * 1000:.6f} ms")

# 3️⃣ SciPy 방식
start = time.perf_counter()
sim_scipy = 1 - distance.cosine(sent_1, sent_2)
end = time.perf_counter()
print(f"[3. SciPy      ] 유사도: {sim_scipy:.6f} | 수행시간: {(end - start) * 1000:.6f} ms")

# 4️⃣ scikit-learn 방식
start = time.perf_counter()
sim_sklearn = pairwise.cosine_similarity(all_sent)
end = time.perf_counter()
print(f"[4. scikit-learn]\n유사도 행렬:\n{sim_sklearn}")
print(f"[4. scikit-learn] 수행시간: {(end - start) * 1000:.6f} ms")
