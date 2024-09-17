# SNLSR
The code of Exploring the Spectral Prior for Hyperspectral Image Super-resolution

Published in IEEE Transactions on Image Processing 24

## Overview :<br>
In recent years, many single hyperspectral image super-resolution methods have emerged to enhance the spatial resolution of hyperspectral images without hardware modification. However, existing methods typically face two significant challenges. First, they struggle to handle the high dimensional nature of hyperspectral data, which often results in high computational complexity and inefficient information utilization. Second, they have not fully leveraged the abundant spectral information in hyperspectral images. To address these challenges, we propose a novel hyperspectral super-resolution network named SNLSR, which transfers the super-resolution problem into the abundance domain. Our SNLSR leverages a spatial preserve decomposition network to estimate the abundance representations of the input hyperspectral image. Notably, the network acknowledges and utilizes the commonly overlooked spatial correlations of hyperspectral images, leading to better reconstruction performance. Then, the estimated low-resolution abundance is super-resolved through a spatial spectral attention network, where the informative features from both spatial and spectral domains are fully exploited. Considering that the hyperspectral image is highly spectrally correlated, we customize a spectral-wise non-local attention module to mine similar pixels along spectral dimension for high-frequency detail recovery. Extensive experiments demonstrate the superiority of our method over other state-of-the-art methods both visually and metrically.
## Training :<br>
```
sh demo.sh
```
## Testing :<br>

## Recommended Environment:<br>

 - [ ] python = 3.9.16
 - [ ] torch = 1.10.0
 - [ ] numpy = 1.23.5
 - [ ] scipy = 1.9.1
 - [ ] scikit-image = 0.17.2

## Contact :<br>
If you have any questions, please feel free to contact me at huq1an@whu.edu.cn

## Citation :<br>
