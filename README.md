# CLAS-pytorch
This is an official implementation of "**Temporal-consistent segmentation of echocardiography with co-learning from appearance and shape**" (MICCAI 2020 early accept, Oral)

[[Paper]](https://www.researchgate.net/publication/342520911_Temporal-consistent_Segmentation_of_Echocardiography_with_Co-learning_from_Appearance_and_Shape)
[[Slide]](https://app.box.com/s/5fkddoh2wa7ygfj80cjmo70s620wi5sr)

# Dataset / Challenge
[Cardiac Acquisitions for Multi-structure Ultrasound Segmentation, CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html)

# Method (CLAS)
![CLAS](https://github.com/HongrongWei/CLAS-Pytorch/blob/main/misc/CLAS.jpg)

# Results (SOTA)

* **Visualization of temporal consistency**

![a2c_im](https://github.com/HongrongWei/CLAS-Pytorch/blob/main/misc/Pat44_A2C_images.gif)
![a2c_seg](https://github.com/HongrongWei/CLAS-Pytorch/blob/main/misc/Pat44_A2C_segmentation.gif)

![a4c_im](https://github.com/HongrongWei/CLAS-Pytorch/blob/main/misc/Pat44_A4C_images.gif)
![a4c_seg](https://github.com/HongrongWei/CLAS-Pytorch/blob/main/misc/Pat44_A4C_segmentation.gif)

* **Cardiac** (endocardium, epicardium, and left atrium) **segmentation**

| Methods       | Endo           |        |             | Epi        |     |            | LA          |      |           |
|:-------------:|:--------------:|:------:|:-----------:|:----------:|:---:|:----------:|:-----------:|:----:|:---------:|
| ED phase[^Note1]      | Dice           | HD[^Note2]     | MAD[^Note3]         | Dice       | HD  | MAD        | Dice        | HD   | MAD       |
| [U-Net](http://camus.creatis.insa-lyon.fr/challenge/#phase/5ca211272691fe0a9dac46d6) | 0.936 | 5.3  | 1.7 | 0.956 | 5.2 | 1.7 | 0.889 | 5.7  | 2.2 |
| [ACNNs](http://camus.creatis.insa-lyon.fr/challenge/#phase/5ca211272691fe0a9dac46d6) | 0.936 | 5.6  | 1.7 | 0.953 | 5.9 | 1.9 | 0.881 | 6.0  | 2.3 |
| CLAS    | 0.947 | 4.6 | 1.4 | 0.961 | 4.8 | 1.5 | 0.902 | 5.2 | 1.9 |
| ES phase[^Note1]  | Dice           | HD    | MAD | Dice           | HD    | MAD    | Dice           | HD           | MAD          |
| [U-Net](http://camus.creatis.insa-lyon.fr/challenge/#phase/5ca211272691fe0a9dac46d6)   | 0.912   | 5.5    | 1.7  | 0.946  | 5.7  | 1.9  | 0.918 | 5.3 | 2.0  |
| [ACNNs](http://camus.creatis.insa-lyon.fr/challenge/#phase/5ca211272691fe0a9dac46d6)   | 0.913   | 5.6    | 1.7  | 0.945  | 5.9  | 2.0  | 0.911 | 5.8 | 2.2  |
| CLAS     | 0.929 | 4.6 | 1.4 | 0.955 | 4.9 | 1.6 | 0.927 | 4.8 | 1.8 |

[^Note1]: end-diastole and end-systole phases
[^Note2]: Hausdorff distance
[^Note3]: Mean absolute distance
      
* **Volumes** (EDV & ESV) and **ejection fraction** (EF) **estimation**

| Methods            | EDV           |          |      | ESV            |          |      | EF             |          |     |
|:------------------:|:-------------:|:--------:|:----:|:--------------:|:--------:|:----:|:--------------:|:--------:|:---:|
|                    | corr[^Note4]          | bias(ml) | std  | corr           | bias(ml) | std  | corr           | bias(\%) | std |
| [U-Net](http://camus.creatis.insa-lyon.fr/challenge/#phase/5ca211272691fe0a9dac46d6)  | 0.926 | 7.2 | 15.6 | 0.960  | 4.4  | 10.2 | 0.845  | 0.1 | 7.3 |
| [ACNNs](http://camus.creatis.insa-lyon.fr/challenge/#phase/5ca211272691fe0a9dac46d6)  | 0.928 | 2.8 | 15.5 | 0.954  | 2.0  | 10.1 | 0.807  | 0.3 | 8.3 |
| CLAS  | 0.958  | -0.7 | 15.1   | 0.979   | -0.0     | 8.4  | 0.926  | -0.1 | 6.7 |

[^Note4]: Pearson Correlation Coefficient

# Citation
Please cite our paper if you find anything helpful:

```
@InProceedings{CLAS,
author={Wei, Hongrong and Cao, Heng and Cao, Yiqin and Zhou, Yongjin and Xue, Wufeng and Ni, Dong and Li, Shuo},
title={Temporal-Consistent Segmentation of Echocardiography with Co-learning from Appearance and Shape},
booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020},
year={2020},
publisher={Springer International Publishing},
pages={623--632},
isbn={978-3-030-59713-9}
}
```

