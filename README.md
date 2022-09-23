# AITM
TensorFlow implementation of Adaptive Information Transfer Multi-task (AITM) framework.  
Code for the paper accepted by KDD21: 
Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising.
[https://arxiv.org/abs/2105.08489]

# Requirement
python==3.6  
tensorflow-gpu==1.10.1  

# Dataset
We use the public Ali-CCP (Alibaba Click and Conversion Prediction) dataset. [https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408].

Please download and unzip the dataset first.

Split the data to train/validation/test files to run the codes directly:
```
python process_public_dataset.py
```



# Example to run the model
```
python AITM.py --embedding_dim 5 --lr 1e-3 --early_stop 1 --lamda 1e-6 --prefix AITM --weight 0.6
```

The instruction of commands has been clearly stated in the codes (see the parse_args function).

# Note
Recently we reformatted the model code as ```AITM_standard.py```. If you want to run the model on multiple tasks (more than two), you can directly pass in the parameter ```--num_tasks``` in ```AITM_standard.py```. But you need to configure the ```config.csv``` file to specify the size of the feature dictionary.

# Reference
If you are interested in the code, please cite our paper:
```
Xi D, Chen Z, Yan P, et al. Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising[C]//Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021: 3745-3755.
```
or in bibtex style:
```
@inproceedings{xi2021modeling,
  title={Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising},
  author={Xi, Dongbo and Chen, Zhen and Yan, Peng and Zhang, Yinger and Zhu, Yongchun and Zhuang, Fuzhen and Chen, Yu},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={3745--3755},
  year={2021}
}
```

# Other unofficial implementations for reference:
## A PyTorch implementation of multi-task recommendation models
[https://github.com/easezyc/Multitask-Recommendation-Library]
## A Pytorch implementation
[https://github.com/adtalos/AITM-torch]

Last Update Date: Sep. 23, 2022 (UTC+8)
