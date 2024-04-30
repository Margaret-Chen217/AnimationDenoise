import pickle as pk
import numpy as np
import os

import denoise
import visualize

root_path = "C:\\Users\\18523\\Desktop\\test\\denoise"

with open(os.path.join(root_path, 'data.pk'), "rb") as fid:
    res_db = pk.load(fid)

def rot2quat(rot):
    """将旋转矩阵转换成四元数"""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rot.reshape(9)
    q_abs = np.array(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ]
    )
    q_abs = np.sqrt(np.maximum(q_abs, 0))

    quat_by_rijk = np.vstack(
        [
            np.array([q_abs[0] ** 2, m21 - m12, m02 - m20, m10 - m01]),
            np.array([m21 - m12, q_abs[1] ** 2, m10 + m01, m02 + m20]),
            np.array([m02 - m20, m10 + m01, q_abs[2] ** 2, m12 + m21]),
            np.array([m10 - m01, m20 + m02, m21 + m12, q_abs[3] ** 2]),
        ]
    )
    flr = 0.1
    quat_candidates = quat_by_rijk / np.maximum(2.0 * q_abs[:, None], 0.1)

    idx = q_abs.argmax(axis=-1)

    quat = quat_candidates[idx]
    return quat


joint_num = 55
frame_num = 60

origin_mat = np.array(res_db["pred_thetas"])
origin_mat = origin_mat.reshape(frame_num, joint_num, 9) 
result_mat = np.zeros((frame_num, joint_num, 4))

for frame in range(frame_num):
    pose = origin_mat[frame]
    for ibone, mrot in enumerate(pose):
        quaternion = rot2quat(mrot)
        result_mat[frame, ibone] = quaternion
# 预处理

# 滑动窗口
smoothed_quaternions = denoise.smooth_quaternions(result_mat, 55,window_size=3)

# butter
butter_quaternions = denoise.butterworth(result_mat, 55)

# fft
fft_quaternions = denoise.fft_smooth(result_mat, frame_num, 55)

# wiener filter
wiener_quaternions = denoise.wiener_smooth(result_mat, 55)

# gaussian filter
gaussian_quaternions = denoise.gaussian_smooth(result_mat, 55)

# 
# visualize.save_result(result_mat, "4_2_0", "origin")
# visualize.save_result(smoothed_quaternions, "4_2_1", "slide")
# visualize.save_result(butter_quaternions, "4_2_2", "butter")
# visualize.save_result(fft_quaternions, "4_2_3", "fft")
# visualize.save_result(wiener_quaternions, "4_2_4", "wiener")
# visualize.save_result(gaussian_quaternions, "4_2_5", "gaussian")

import evaluate
result = evaluate.mse_quaternions(result_mat, smoothed_quaternions)
print("MSE Slide:", result)

result = evaluate.mse_quaternions(result_mat, butter_quaternions)
print("MSE butter:", result)

result = evaluate.mse_quaternions(result_mat, fft_quaternions)
print("MSE fft:", result)

result = evaluate.mse_quaternions(result_mat, wiener_quaternions)
print("MSE wiener:", result)

result = evaluate.mse_quaternions(result_mat, gaussian_quaternions)
print("MSE gaussian:", result)


result = evaluate.psnr_quaternions(result_mat, smoothed_quaternions)
print("PSNR Slide:", result)

result = evaluate.psnr_quaternions(result_mat, butter_quaternions)
print("PSNR butter:", result)

result = evaluate.psnr_quaternions(result_mat, fft_quaternions)
print("PSNR fft:", result)

result = evaluate.psnr_quaternions(result_mat, wiener_quaternions)
print("PSNR wiener:", result)

result = evaluate.psnr_quaternions(result_mat, gaussian_quaternions)
print("PSNR gaussian:", result)
