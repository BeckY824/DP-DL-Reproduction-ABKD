<div align="center">
  <img src="https://github.com/user-attachments/assets/9e2d843c-4d77-4357-8691-713872906d23" alt="ABKD", width="250px">
  <h1>ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence</h1>
</div>

<!-- <a href="https://arxiv.org/abs/2402.03898"><img src="https://img.shields.io/badge/Paper-arXiv:2402.03898-Green"></a>
<a href=#bibtex><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a> -->

This repository is forked from the original implementation and serves as a reproduction of the method. The original paper is available [here](https://arxiv.org/abs/2505.04560).

We only reproduced the **standard classification task**, and further extended the experiments to include **CIFAR-100-LT**.




## Table of Contents
- [Standard Classification Task](#Standard-Classification-Task)
- [Extended Task for CIFAR100-LT](#Extended-Task-for-CIFAR100-LT)
## Standard Classification Task

Please make sure you are in the `standard_classification` directory:

```
cd standard_classification
```

Fetch the pretrained teacher models by:
```
bash scripts/fetch_pretrained_teachers.sh
```
which will download and save the models to `save/models`

### Running

- To run vanilla KD:
```
bash train_kd.sh 
```
- To calibrate the loss function used in vanilla KD and obtain our proposed **ABKD**:
```
bash train_ab.sh 
  1.1 \  # start_alpha_beta: Starting value of (alpha + beta)
  1.1 \  # end_alpha_beta: Ending value of (alpha + beta)
  0.8 \   # start_alpha
  0.8 \   # end_alpha
  resnet56 \   # teacher_model
  resnet20 \   # student_model
  0 \   # gpu_id
  32    # b (weight for distillation loss)
```
  
- To run other baselines (e.g., LSD):
```
bash train_ls.sh 1.0 1.0 0 0 resnet56 resnet20 \
  0 # gpu id
```

- To calibrate the loss function of LSD and obtain ABLSD:
```
bash train_ls.sh 1.2 1.2 0.9 0.9 resnet56 resnet20 0
```

The resulting log file of an experiment recording test accuracy after each epoch is saved in './save'.

## Extended Task for CIFAR-100-LT

## BibTeX
If you find this repo useful for your research, please consider citing their paper:

```
@article{wang2025abkd,
  title={ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via $$\backslash$alpha $-$$\backslash$beta $-Divergence},
  author={Wang, Guanghui and Yang, Zhiyong and Wang, Zitai and Wang, Shi and Xu, Qianqian and Huang, Qingming},
  journal={arXiv preprint arXiv:2505.04560},
  year={2025}
}
```

## Contact
- Guanghui Wang: guanghui6691@gmail.com

## Acknowledgements

Our code is based on [DISTILLM](https://github.com/jongwooko/distillm), [PromptKD](https://github.com/zhengli97/PromptKD/blob/main/README.md), [TTM](https://github.com/zkxufo/TTM) and [MINILLM](https://github.com/microsoft/LMOps/tree/main/minillm). We thank the authors for releasing their code.
