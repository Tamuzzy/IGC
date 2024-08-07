# IGC
This is the demo implementation of our paper: [Learning Flexible Time-windowed Granger Causality Integrating Heterogeneous Interventional Time Series Data](https://arxiv.org/abs/2406.10419).


## Introduction
Granger causality, commonly used for inferring causal structures from time series data, has been adopted in widespread applications across various fields due to its intuitive explainability and high compatibility with emerging deep neural network prediction models. To alleviate challenges in better deciphering causal structures unambiguously from time series, the use of interventional data has become a practical approach. However, existing methods have yet to be explored in the context of imperfect interventions with unknown targets, which are more common and often more beneficial in a wide range of real-world applications. Additionally, the identifiability issues of Granger causality with unknown interventional targets in complex network models remain unsolved. Our work presents a theoretically-grounded method that infers Granger causal structure and identifies unknown targets by leveraging heterogeneous interventional time series data. We further illustrate that learning Granger causal structure and recovering interventional targets can mutually promote each other. Comparative experiments demonstrate that our method outperforms several robust baseline methods in learning Granger causal structure from interventional time series data.


## Citation
If you find it useful, please cite our paper. Thank you!
```
@article{zhang2024learning,
  title={Learning Flexible Time-windowed Granger Causality Integrating Heterogeneous Interventional Time Series Data},
  author={Zhang, Ziyi and Ren, Shaogang and Qian, Xiaoning and Duffield, Nick},
  journal={arXiv preprint arXiv:2406.10419},
  year={2024}
}
```
