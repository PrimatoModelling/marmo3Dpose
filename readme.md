# Model card for Marmo3Dpose

## Model details
This tool is designed to process multiple videos from various viewpoints and generate 3D pose estimations for groups of marmosets. It is especially useful for analyzing the behavior of freely moving marmosets in laboratory environments. The tool can be applied in areas such as biomedical, ethological, applied animal sciences and neuroscience research. The training dataset for this tool comprises over 56,000 annotations, enabling accurate 3D pose estimation of marmosets.

- Analytic pipeline combining multiple CNNs and 3D reconstruction utilities. 
- Pretrained network optimized for 3D pose estimation of marmoset families, which include a father, a mother, and an infant. The pre-trained model outputs time series data of 3D positions for eyes, nose, shoulders, elbows, wrists, hips, knees, ankles. The dataset is available [here](https://doi.org/10.5281/zenodo.11180331). 

### Reference 
If you use the code or data, please cite us:   
[Deciphering social traits and pathophysiological conditions from natural behaviors in common marmosets](https://www.biorxiv.org/content/10.1101/2023.10.16.561623v1)   
by Takaaki Kaneko and Jumpei Matsumoto et al. ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.16.561623v1))

### Demo 
Please see [howToStart.md](howToStart.md)  

### License
- The tools and data are made available under the Apache 2.0 license.   

### Acknowledgment 
Our work builds on the previous significant contributions:
- [anipose](https://github.com/lambdaloop/anipose)  
  Karashchuk, P., Rupp, K.L., Dickinson, E.S., Walling-Bell, S., Sanders, E., Azim, E., Brunton, B.W., and Tuthill, J.C. (2021). Anipose: A toolkit for robust markerless 3D pose estimation. Cell Rep. 36, 109730.
- [mvpose](https://github.com/zju3dv/mvpose)  
  Dong, J., Jiang, W., Huang, Q., Bao, H., and Zhou, X. (2019). Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views. arXiv:1901.04111.
- [openmmlab](https://github.com/open-mmlab)  
  Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., et al. (2019). MMDetection: Open MMLab Detection Toolbox and Benchmark. arXiv:1906.07155.

## Intended use
- This tool is designed to acquire 3D pose estimation data of multiple marmosets for behavioral analysis.
- While the model itself only provides 3D coordinate information, when combined with appropriate downstream analyses, it can be adapted to a variety of neuroscience experiments including ethological analysis, cognitive function, and phenotypic analysis of disease models.
- It has also been confirmed to work with macaques and can be adapted for use with other animals, albeit with some adjustments.

### Out-of-scope use cases
- This tool is optimal for analyzing behaviors that involve whole-body movements, particularly those involving limbs and the head. However, it is not suitable for analyses that require details beyond keypoints, such as subtle facial expressions or piloerection.

## Factors
- This tool is not suitable for marmosets under one month old that are always riding on their parents' backs.
- Accurate individual identification enhances the precision of 3D estimation. When using color tags or similar methods for individual distinction, it is important to note that accuracy might decrease over time due to dirt or wear.
- It is preferable for each keypoint to be visible in at least three cameras. As the number of subjects or obstacles increases, it may be necessary to increase the number of cameras to maintain accuracy.

## Metrics
- The animal detection and identification in 3D space were 99.3% and 98.8% in precision and recall, respectively. 
- The geometric error in pose estimation at each keypoint ranged between 4.86 and 17.0 mm.
- For more details, please refer to the paper.

## Training and Evaluation data
- The data used for training and evaluation is included in this toolkit.
- Annotations were made for 56,103 instances across 29 animals, captured in the form of 2D images. Since one image can include multiple animals, there are a total of 23,052 images.

## Quantitative analyses
The utility of the 3D pose data cannot be fully assessed by the metrics mentioned above. Therefore, the practical utility of this model has been validated through a series of real biological experiments:

- The model was capable of automatically detecting the differential contributions to parenting behaviors between mothers and fathers.
- By integrating pose estimation with state estimation, the model could uncover the complex and adaptive social cognitive functions of marmosets.
- The model elucidated the progression of symptoms in a Parkinson's disease model animal over a year, using unsupervised clustering methods, without any prior assumptions.

## Ethical considerations
All procedures for the use and experiments of common marmosets were approved by the Animal Welfare and Animal Care Committee of the Center for the Evolutionally Origins of the Human Behavior, Kyoto University, followed by the Guidelines for Care and Use of Nonhuman Primates established by the same institution.

## Caveats and recommendations
The model has various hyperparameters optimized for our specific experimental environment. The training images are also environment-dependent. Consequently, one should not expect the model to function with the same level of accuracy immediately in a new setting. Fine-tuning of the pose estimation, ID recognition, and detection network is essential. Additionally, camera calibration parameters need to be specifically obtained for the new site. For further information, please refer to the paper.





