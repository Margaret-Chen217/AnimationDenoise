import numpy as np

def quaternion_distance(q1, q2):
    # 计算两个四元数之间的角度差
    dot_product = np.abs(np.sum(np.multiply(q1, q2), axis=-1))
    dot_product = np.clip(dot_product, -1.0, 1.0)  # 避免计算误差导致的范围外值
    theta = 2 * np.arccos(dot_product)
    return theta

def mse_quaternions(q_true, q_pred):
    # 确保q_true和q_pred的shape相同
    if q_true.shape != q_pred.shape:
        raise ValueError("The shape of q_true and q_pred must be the same.")

    # 计算每对四元数的角度差的平方
    squared_errors = np.square(quaternion_distance(q_true, q_pred))

    # 计算均方误差
    mse = np.mean(squared_errors)
    return mse

def psnr_quaternions(q_true, q_pred, max_val=np.pi):
    mse = mse_quaternions(q_true, q_pred)
    if mse == 0:
        return float('inf')  # 避免对零进行除法
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr