import numpy as np
import matplotlib.pyplot as plt


def save_result(matrix, filename, title):
    
    frame_num = matrix.shape[0]
    joint_num = matrix.shape[1]
    time_steps = np.arange(frame_num)

    # 绘制所有关节的四元数曲线
    fig, axs = plt.subplots(4, 1, figsize=(12, 8))

    for component_index in range(4):
        ax = axs[component_index]
        
        for joint_index in range(10):
            quaternions = matrix[:, joint_index, :]
            ax.plot(time_steps, quaternions[:, component_index], label=f'Joint {joint_index}')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'Quaternion - Q{component_index}')
        ax.set_title(f'Q{component_index}')
        ax.legend().remove()
    # 调整子图间距
    plt.tight_layout()
    # 添加图片标题
    plt.suptitle(title, fontsize=20)
    

    # 调整子图和标题之间的间距
    plt.subplots_adjust(top=0.9)
    # 保存图形为图片文件
    import os
    output_path = os.path.join("img", filename)
    plt.savefig(output_path)

    print(f'图形已保存为：{output_path}')
