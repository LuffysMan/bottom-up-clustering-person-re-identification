# A Bottom-Up Clustering Approach to Unsupervised Person Re-identification

This repo is a pytorch re-implementation for paper [[Link]](https://vana77.github.io/vana77.github.io/images/AAAI19.pdf). 
The original version is [[vana77]](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification). 
Some basic libraries also references [[michuanhaohao]](https://github.com/michuanhaohao/reid-strong-baseline). Such as data processing.


The updated performances on the four re-ID datasets are listed below:

|               | rank-1        | rank-5    | rank-10   | mAP     |
| ----------    | :-----------: | :-----------: |:-----------:  | :-----------: |
| Market-1501   | 44.5          | ----      |----       | 19.6     |

## Preparation
### Dependencies
- Python 3.8.5
- PyTorch (version >= 1.70)
- yacs

### Download datasets 
The dataset structure used in this repo is slightly different from it in the original code. But the content should be the same. I'll soon post links for dataset download here. 

## Usage

```shell
sh method_BUC/z_end2end_default_settings.sh
```
I utilize 1 GTX-1080TI GPU for training on image-based datasets.

## Citation
Please refer to [[vana77]](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification).

## ToDos
- To improve performance on market1501 since it is far behind the reported performance.
- To add results on dukemtmc-reid and other commonly used re-id datasets.
- To further refactor the code and project structure for symplicity.





