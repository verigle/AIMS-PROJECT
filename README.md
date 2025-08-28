# AI For Science Master project 
---

[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](#)
[![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?logo=visual-studio-code&logoColor=white)](#)
[![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)](#)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](#)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-4285F4?logo=googlecloud&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#)
![Awesome](https://img.shields.io/badge/Awesome-ffd700?logo=awesome&logoColor=black)

---

![Project Screenshot](images/aurora.jpg)


# Overview

Accurate weather forecasts are critical for climate resilience, but traditional numerical weather prediction (NWP) systems are too costly and resource-intensive for many African regions. AI weather prediction (AIWP) models offer a faster and more affordable alternative, but they usually struggle in Africa due to limited meteorological infrastructure and scarce local data.

This project explores fine-tuning a pretrained AI foundation model (Aurora) on South African weather data. By adapting state-of-the-art models to regional conditions, we aim to deliver affordable, locally adapted forecasts that can support communities, researchers, and policymakers across the continent.

Key goals:

- Make weather forecasting more accessible in low-resource settings

- Improve predictive accuracy in Africa through regional fine-tuning

- Contribute to climate resilience with AI-driven solutions


# How to run?
## Clone the repository

```bash
git clone https://github.com/koomited/AIMS-PROJECT
```
## Create a virtual environment and activate it

```bash
python3 -m venv venv
source venv/bin/activate
```


## Install the requirements

```bash
pip install -r requirements.txt
```
Make sure you download the checkpoint of the pretrained models you need, not the fine tuned one if you want to train, first from Hugging Face and save them in a folder named `model` in your root directory.
You can explore how to do that with this notebook [`notebooks/model_surgery.ipynb`](notebooks/model_surgery.ipynb)
Note that we download the ones we need, the small and large 0.25° resolution models. You can donwload other version of the model from [Huggin Face](https://huggingface.co/microsoft/aurora/tree/main)




To run the code in this project a linux environment particulary ubuntu is required. Also notice that we build the whole project on Google Gloud Plateform (GCP). Therefore you may need to check out the correspondence of some codes if you are running the code on your local machine.

## Cross-region check of the model perfomance: South Africa, Europe, USA
In case you are interested in checking the model care about performance over different regions, you can start with

[`scripts/rmses_grid_sp_sa_vs_usa_eu.py`](scripts/rmses_grid_sp_sa_vs_usa_eu.py) where we did for the regions above. You can change those regions in the scripts if you want and run it with the code below. You can also change the model you are interested in.

```bash
nohup python rmses_grid_sp_sa_vs_usa_eu.py > rmses_grid_sp_sa_vs_usa_eu.log 2>&1 &
```
The log files keep track of the progress. You can open and check.
You can find the plots in [`report/evaluation/rmses_grid/pretrained_small`](report/evaluation/rmses_grid/pretrained_small).
## Training

The main code for training can be found in  [`scripts/training_on_hrest0_wampln.py`](scripts/training_on_hrest0_wampln.py).

To run it, you need to pay attention to some few things:
- Make sure you change the code on the line 72 to the timeframe you want to train on. Remember that we are training on HREST TO dataset and don't exide its time frame. 

```python
start_time, end_time = '2019-01-01', '2021-12-31' 
```

Also remember we are training the small version of the model. In case you want to change this, you can do it from line 39 to 54
```python
model = AuroraSmall(
    use_lora=False,  .
)

model = full_linear_layer_lora(model, lora_r = 16, lora_alpha = 4)
checkpoint = torch.load('../model/training/hrest0/wampln/checkpoint_epoch_13.pth')

model.load_state_dict(checkpoint['model_state_dict'])
```
Because of the computational demand of the traning, we save each the checkpoint of each epoch in this folder [`model/training/hrest0/wampln`](model/training/hrest0/wampln). You can also resume training from any checkpoint by changing the python code just above.

Make sure you are in the folde [`scripts`](scripts) in the terminal and use the following command to start the training.

```bash
nohup python training_on_hrest0_wampln.py > training_on_hrest0_wampln.log 2>&1 &
```

## Evaluation
Here we create scorecard where we can compare two different models over South Africa. Example:
### Fine tuned small model against pretrained small models
```bash
nohup python evaluation_run_wampln_smalftvs_pretrained_sa.py > evaluation_run_wampln_smalftvs_pretrained_sa.log 2>&1 &
```
The plots are saved in [`report/evaluation/wampln`](report/evaluation/wampln).

Similary to the perfromance check accross regions using the pretrained model, you can the same thing for your fin tuned models. You can find an example in [`scripts/rmses_grid_ft_sa_vs_usa_eu.py`](scripts/rmses_grid_ft_sa_vs_usa_eu.py). You ran this script exactely the same way we did and you can check where you plots are save yourself b y looking at the saving path in the file oor just ckeck the [`report/evaluation/rmses_grid/fine_tuned_small`](report/evaluation/rmses_grid/fine_tuned_small).



##  Future works
- Training on local weather station data
- Training on more data
- Investigate other fine-tuning techniques

---

👨‍💻 Crafted with ❤️ by **Koomi Toussaint AMOUSSOUVI**

📫 Reach out to me on [LinkedIn](https://www.linkedin.com/in/koomi-toussaint-amoussouvi-87b923201/) 
