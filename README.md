# AITM
TensorFlow implementation of Adaptive Information Transfer Multi-task (AITM) framework.  
Code for the paper submitted to KDD21: 
Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising.


# Requirement
python==3.6  
tensorflow-gpu==1.10.0

# Dataset
We use the public Ali-CCP (Alibaba Click and Conversion Prediction) dataset. [https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408].

Please download and unzip the dataset first.

Split the data to train/validation/test files to run the codes directly:
```
python process_public_dataset.py
```



# Example to run the model
```
python AITM.py --embedded_dim 5 --lr 1e-3 --early_stop 1 --lamda 1e-6 --prefix AITM --weight 0.6
```

The instruction of commands has been clearly stated in the codes (see the parse_args function).

Last Update Date: Jan. 25, 2021 (UTC+8)
