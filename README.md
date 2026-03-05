# LU-SSD: A Lightweight U-Shaped State Space Duality Network for Precipitation Downscaling
Qingguo Su, Di Zhang, Xinjie Shi, Kefeng Deng, Kaijun Ren
# About
Despite significant progress, the high computational cost makes existing precipitation downscaling methods difficult to deploy on resource-constrained devices. To address this challenge, we propose the LU-SSD, which jointly enhances the physical fidelity and spatial resolution of total precipitation from ERA5 through a synergistic strategy of "constructing foundational skeletons" and " decorating with detailed structures" in both spatial and frequency domains. Specifically, we integrate 20 key meteorological variables, utilizing improved radar observations as ground truth to upscale the precipitation by a factor of 8. First, we analyze the key factors contributing to the computational complexity of State Space Duality Networks (SSDs) and accordingly propose a Separable Channel State Space Duality Module (SC-SSD). A lightweight U-shaped backbone network stacked with SC-SSDs effectively models long-range spatial dependencies among meteorological variables at a cheap computational cost, accurately establishing the foundational skeletons of precipitation areas and rain bands. Second, multi-scale convolutional modules are incorporated to capture local spatial details, thereby reconstructing the fine-scale structures. Furthermore, we analyze systematic biases between input and ground truth in the frequency domain, and design a Precipitation Inverse-Frequency Attention Module (PIAM), which corrects low-frequency component biases and reconstructs extreme precipitation events. Finally, a Region-aware Intensity Conservation Module (RICM) is proposed to embed the intensity conservation laws into the network, ensuring physical consistency. Extensive experiments demonstrate that the LU-SSD achieves state-of-the-art performance. Notably, it requires only 678 MB of GPU memory during inference, enabling efficient deployment on resource-constrained devices and offering a viable pathway toward high-resolution meteorological services.
# Contents

1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)



## Training
Used training sets can be downloaded as follows:
### Training Datasets
1. [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. [Flickr2K](https://www.kaggle.com/datasets/hliang001/flickr2k)

### Train
If you want to run a different configuration, modify `configs\config_setting.py`.
```
python train.py --num_workers=8 --batch_size=64
```
## Testing
Used testing sets can be downloaded as follows:

### Testing Datasets
1. [Set5](https://paperswithcode.com/dataset/set5)
2. [Set14](https://paperswithcode.com/dataset/set14)
3. [BSD100](https://paperswithcode.com/dataset/bsd100)
4. [Urban100](https://paperswithcode.com/dataset/urban100)
5. [Manga109](https://paperswithcode.com/dataset/manga109)
### Test
```
python test.py
```
## Results
<p align="center">
  <img width="900" src="https://github.com/SuMuzi/LU-M2SR/results/results.png">
</p>
