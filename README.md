

# CLiD-MI (**C**onditional **Li**kelihood **D**iscrepancy)
The code of the following paper will be released in this repository:

[**Membership Inference on Text-to-image Diffusion Models via Conditional Likelihood Discrepancy**](https://arxiv.org/abs/2405.14800) **(NeurIPS 2024)**.


## Usage - Finetuning Setting

1. **Fine-tuning Target and Shadow Models**
   - Use `ft_mia.sh` script to fine-tune the target and shadow models on two different training sets ([COCO_MIA_Finetuning Data](https://huggingface.co/datasets/zsf/COCO_MIA_ori_split1)).

2. **Performing Membership Inference**
   - Utilize the following scripts to conduct membership inference on the fine-tuned models:
     - [ ]  `mia_Loss.py` 
     - [x]  `mia_pfami.py`
     - [ ]  `mia_SEC_PIA.py`
     - [x]  `mia_CLiD_impt.py` (or `mia_CLiD_clip.py`): CLiD with the reduction methods of _Importance clipping_ and _Simply Clipping_ 

3. **Evaluation**
   - Use the `cal_xx.py` functions to compute the metrics of the following methods:
     - Baselines: `cal_baselines.py`
     - Clidth: `cal_clid_th.py`
     - Clidvec: `cal_clid_xgb.py`
     
4. **Validation results with MS-COCO Dataset under real-world training settings**:
   - We additionally provide the intermediate results of the MS-COCO dataset in Sec. 4.2 for validation.
   - These results are obtained under real-world training settings.


##  Usage - Pretraining Setting
- [ ] To be added.



## Citation
If you find this project useful in your research, please consider citing our paper:
```
@article{zhai2024membership,
  title={Membership inference on text-to-image diffusion models via conditional likelihood discrepancy},
  author={Zhai, Shengfang and Chen, Huanran and Dong, Yinpeng and Li, Jiajun and Shen, Qingni and Gao, Yansong and Su, Hang and Liu, Yang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={74122--74146},
  year={2024}
}
```
