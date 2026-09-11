# PPT 优化算法的实现与复现

实现对应 PPT 第 6–7 页：谱残差显著性检测、显著图归一化并二值化，以及
`e_new(x,y) = e_grad(x,y) + lambda * M_sal(x,y)`，默认 `lambda=1e11`。
保留 `backward` 和 `forward`，新增 `energy_mode="saliency"`。

## 运行你的测试图片

在项目根目录执行（本项目已有可用的 `.venv`）：

```powershell
.\.venv\Scripts\python.exe -m example.saliency_demo --input "C:/Users/robin/Downloads/Snipaste_2026-09-10_01-06-46.png"
```

默认宽度缩至 70%，高度不变；可用 `--width-ratio 0.6` 调整。
`--threshold 0.2` 控制二值化，阈值越低，保护范围越大。
`--weight 1e11` 控制保护能量。
输出在 `outputs/saliency_demo`：原图、基础算法结果、改进结果、显著图、
二值掩膜、保护区域叠加图、对比图以及记录参数与耗时的 `run.json`。
所有对比结果在相同目标尺寸下生成，不使用人工人物标注。

## Python 调用

```python
import numpy as np
from PIL import Image
import seam_carving

src = np.array(Image.open("your_image.png").convert("RGB"))
h, w = src.shape[:2]
dst = seam_carving.resize(
    src, (round(w * 0.7), h),
    energy_mode="saliency",
    saliency_threshold=0.2,
    saliency_weight=1e11,
)
Image.fromarray(dst).save("result.png")
```

`spectral_residual(src)` 可独立计算显著图；`saliency_mask(S, threshold)`
可独立生成保护掩膜。`resize(..., saliency_map=S)` 可复用同一张归一化显著图。

## 实现细节与假设

1. 转灰度，双线性缩放至 64×64，做 FFT。
2. `L = log(max(abs(FFT), 1e-12))`，计算 `R = L - mean_filter_3x3(L)`。
3. 保留原始相位，计算 `abs(IFFT(exp(R + i*phase)))**2`。
4. 用 sigma=2.5 的高斯滤波平滑，双线性恢复原始大小，min-max 归一化。
5. 以 `S > 0.2` 二值化，加入 1e11 能量后进行动态规划。

PPT 明确了方法和 lambda，但没有明确分析尺寸、滤波参数、阈值、
缩放比例或显著图更新频率。上述参数是本实现的默认选择，不应视作 PPT
已有设定。显著图仅在原图上计算一次，其能量图随接缝同步删除、插入、转置，
避免每次重算导致保护区域漂移。基础梯度能量仍按现有算法更新。

补全了原 `carve.py` 中两个 `_ # Fill in codes here` 的动态规划占位。
累计代价改用 float64，以减轻叠加高额保护能量时的精度损失。
改进模式支持宽高缩小、放大和两种处理顺序；不与 `drop_mask` 同时使用，
因为删除目标与高能量保护可能冲突。

谱残差检测的是视觉显著性，不是语义人物分割，保护掩膜未必覆盖整个人物。
1e11 是有限惩罚：如果目标尺寸过小，所有路径都经过保护区域，仍会删除其中像素。
因此应结合 `protection_overlay.png` 与实际输出判断效果，不能保证任意图片和比例
都能无失真地保留主体。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest test -q -p no:cacheprovider
```

新增测试覆盖小图穷举最优接缝、平坦图、显著图归一化、人工合成低纹理主体的
保护、零权重与基础算法一致、宽高放缩、两种处理顺序和非法参数。

## 本次提供图片的结果

原图为 976×841，本次缩至 683×841（宽度保留 70%），阈值 0.2，
lambda=1e11。显著掩膜覆盖约 85.77% 的像素，其中包括大量树林，
而衣服内部的显著性较低。实际结果中脸部比基础版保留得更好，
但躯干仍被明显压缩，未复现 PPT 第 8 页展示的完整人物保护效果。
这个结果反映了当前谱残差方案在该图上的限制；不能将它报告为无失真的成功案例。
若要完整保护人物，后续可在此能量框架中使用人物分割掩膜，但那属于
PPT 所述谱残差方法之外的额外改进。
