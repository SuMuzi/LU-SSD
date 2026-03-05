# LU-SSD: A Lightweight U-Shaped State Space Duality Network for Precipitation Downscaling
Qingguo Su, Di Zhang, Xinjie Shi, Kefeng Deng, Kaijun Ren
# About
Despite significant progress, the high computational cost makes existing precipitation downscaling methods difficult to deploy on resource-constrained devices. To address this challenge, we propose the LU-SSD, which jointly enhances the physical fidelity and spatial resolution of total precipitation from ERA5 through a synergistic strategy of "constructing foundational skeletons" and " decorating with detailed structures" in both spatial and frequency domains. Specifically, we integrate 20 key meteorological variables, utilizing improved radar observations as ground truth to upscale the precipitation by a factor of 8. First, we analyze the key factors contributing to the computational complexity of State Space Duality Networks (SSDs) and accordingly propose a Separable Channel State Space Duality Module (SC-SSD). A lightweight U-shaped backbone network stacked with SC-SSDs effectively models long-range spatial dependencies among meteorological variables at a cheap computational cost, accurately establishing the foundational skeletons of precipitation areas and rain bands. Second, multi-scale convolutional modules are incorporated to capture local spatial details, thereby reconstructing the fine-scale structures. Furthermore, we analyze systematic biases between input and ground truth in the frequency domain, and design a Precipitation Inverse-Frequency Attention Module (PIAM), which corrects low-frequency component biases and reconstructs extreme precipitation events. Finally, a Region-aware Intensity Conservation Module (RICM) is proposed to embed the intensity conservation laws into the network, ensuring physical consistency. Extensive experiments demonstrate that the LU-SSD achieves state-of-the-art performance. Notably, it requires only 678 MB of GPU memory during inference, enabling efficient deployment on resource-constrained devices and offering a viable pathway toward high-resolution meteorological services.
### Environment
Python 3.8+, PyTorch 2.1 and Ubuntu or CenterOS.

# Test
```
python main_test_example_era5.py
```
# Demo 

<table>
  <tr>
    <th width="50%">Super Typhoon Chanthu (No.2114)</th>
    <th width="50%">Severe Typhoon In-Fa (No.2106)</th>
  </tr>
  <tr>
    <td>
      <img src="test/img/LU-SSD/all/Chanthu_LU-SSD.gif" width="100%" />
    </td>
    <td>
      ![In-fa](test/img/LU-SSD/all/In-fa_LU-SSD.gif)
    </td>
  </tr>
</table>
