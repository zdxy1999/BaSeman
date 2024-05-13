## The Implementation of BaSeman

## download data ##
[[click to download LSCIDMR-V2]](https://pan.baidu.com/s/1QUNNjWp_sKiFXE5cuZvc4w?pwd=9999)

fetch code: ```9999```
## Environment Prepare ##
#### pytorch version ####
```
torchvision version: torchvision-0.11.0%2Bcu111-cp38-cp38-linux_x86_64.whl
torch version: torch-1.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl
```
#### other dependencies ####
```
pip install requirements.txt
```
## Inference Model ##
1. dowload trained model
[[click to download trained model]](https://pan.baidu.com/s/1J7wHYhlOJ9Z5Mb8Lwetarw?pwd=9999)
    fetch code: ```9999```
2. move the downloaded folder```/final_result``` into ```/test_ground_1/results```
3. run inference
```
python main.py --model=s2net --saved_model_name=final_result --inference
```

## Train new model ##
#### run training

```
python main.py --dataset=LSCIDMR_16c\
    --model=s2net\
    --optim=adam\
    --dropout=0
```
## Citing ##

```bibtex

```