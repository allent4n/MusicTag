# Auto-Music-Tag-Generation :notes: :guitar: :headphones:	

## We proposed a new model (***CNN-LSTM Autoencoder for Musci Tagging, CLSTMA-MT***) that can Automatically generating music tags

#### To visualize the performance of our CLSTMA-MT model, you can go to this :point_right: [colab link](https://colab.research.google.com/drive/10hhthqFWaMbe_oWFYN2nw12uTNYBAYgc?usp=sharing) :point_left:	

### OR

#### *Clone this repo and follow the steps* to play with this project. :point_down:	:point_down:	

## Our proposed model (CLSTMA-MT) structure:
* **Input**: Music data --> Spectrogram data
* **Encoder**: one CNN layer and one LSTM layer (Spectrogram data --> latent vector)
* **Decoder**: LSTM (latent vector --> Text vector)
* **Output**: Text vector --> Music aspect text

<img width="1037" alt="Screenshot 2023-05-24 at 19 04 12" src="https://github.com/allent4n/auto-music-tag/assets/78404109/59caaaec-58dc-4d35-85b8-f6be93269ccf">

## Our baseline models are RNN-based Autoencoder, LSTM-based Autoencoder, GRU-based Autoencoder, CNN-RNN-based Autoencoder and CNN-GRU-based Autoencoder

### The model performance results are shown below:

Baseline:

|  Models | LSTM |  GRU  |  RNN  | CNN_RNN | CNN_GRU |
| --- |---| --- | ---| --- | --- |
|   Loss  | 4.08 | 4.42  |  4.30 |   3.8   |  3.77   |


***CLSTMA-MT***:

***Loss: 2.89***


## Clone this repo:
``` 
git clone https://github.com/allent4n/auto-music-tag
```

## Dataset (1000 of music data)

Given the MusicCaps data from this link (https://www.kaggle.com/datasets/googleai/musiccaps), a subset of music data (1000 music data) has been sampled for our baseline model training, the sampled music data can be downloaded from:

```
https://liveln-my.sharepoint.com/:f:/g/personal/allentan_ln_hk/ErUL0UhQgHBAtn8O16FZvUwBfoET5grtnqQnZqHYY9rN7Q?e=ePqhL2
```
* The above link contains **1000 music data samples**. You can download all music data or just 100 music data for training


## Environment
Run the following codes to build a **virtual environment** for this project:
```
cd auto-music-tag
mkdir venv
virtualenv -p python3 venv/
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation
* place go to the auto-music-tag folder and generate a new folder(**1k_data**), you can do it by running the following codes:
```
cd auto-music-tag
mkdir 1k_data
```
* Then put all your downloaded music data into the **1k_data** folder.


## CLSTMA-MT Model (main.py)
### Train the CLSTMA-MT model

#### You can either download the trained model parameters from the following link or train your own model:

## For training
* run the following code to train the  model
```
python main.py
```

## For testing with the CLSTMA-MT model
* Download the trained model (1000 music data and 200 epochs), and trained wordtokenizer
```
https://liveln-my.sharepoint.com/:f:/g/personal/allentan_ln_hk/ErT8ToNqyuxIkSkpcn3b1e4BB3ma9E2bD71R1hiTYLtvhw?e=i7gPNT
```
* Build a **result** folder to keep all downloaded files 
```
mkdir result
```
* Then run this code to visualize the possible tags for the testing music
```
python main.py --mode "test"
```

## Baseline Models (baseline.py)
### Train the baseline models

#### You can either download the trained model parameters from the following link or train your own model:

## For training
* Run the following code to train the baseline models (**RNN as example**)
* You can also train other baselines models (gru, lstm, cnn_rnn and cnn_gru) by replacing the rnn in the following code.
```
python baseline.py --model "rnn"
```

## For testing with the baseline models
* Download the trained baseline models (1000 music data and 200 epochs), and trained wordtokenizer
```
https://liveln-my.sharepoint.com/:f:/g/personal/allentan_ln_hk/EhDl5p2Y_8JGig4nsNsXvjwB_-NRccEsWqHVSofm1KURxQ?e=T0Ate7
```
* Run this code to visualize the possible tags for the testing music data
```
python baseline.py --model "rnn" --mode "test"
```
# Enjoy :clinking_glasses:	:clinking_glasses:	
