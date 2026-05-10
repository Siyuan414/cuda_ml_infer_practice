import numpy as np

np.random.seed(0)
X = np.random.randn(1024,512)

X[:,[3,17,128]] *= 100 

#per tensor quantization
def quantize_tensor(x):
    scale = np.max(np.abs(x)) / 127
    q = np.round(x / scale)
    return q, scale

def dequantize(q, scale):
    return q * scale

q, s = quantize_tensor(X)
X_hat = dequantize(q, s)

error_tensor = np.mean((X - X_hat)**2)

#per channel quantization
def quantize_per_channel(x):
    scale = np.max(np.abs(x), axis=0) / 127
    q = np.round(x / scale)
    return q, scale
q_c, s_c = quantize_per_channel(X)
X_hat_c = q_c * s_c
error_channel = np.mean((X - X_hat_c)**2)

print(f"Per tensor quantization error: {error_tensor}")
print(f"Per channel quantization error: {error_channel}")

import matplotlib.pyplot as plt

plt.hist(X.flatten(), bins=100)
plt.title("Activation Distribution with Outliers")
plt.show()