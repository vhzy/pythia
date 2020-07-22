

Pythia is a modular framework for vision and language multimodal research. Built on top
of PyTorch, it features:

## Citation

If you use Pythia in your work, please cite:

```
@inproceedings{singh2019TowardsVM,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

and

```
@inproceedings{singh2018pythia,
  title={Pythia-a platform for vision \& language research},
  author={Singh, Amanpreet and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  booktitle={SysML Workshop, NeurIPS},
  volume={2018},
  year={2018}
}
```

## Getting Started

First install the repo using

```
git clone https://github.com/facebookresearch/pythia ~/pythia

# You can also create your own conda environment and then enter this step
cd ~/pythia
python setup.py develop
```

Now, Pythia should be ready to use. Follow steps in specific sections to start training
your own models using Pythia.


## Data

Default configuration assume that all of the data is present in the `data` folder inside `pythia` folder.

Depending on which dataset you are planning to use download the feature and imdb (image database) data for that particular dataset using
the links in the table (_right click -> copy link address_). Feature data has been extracted out from Detectron and are used in the
reference models. Example below shows the sample commands to be run, once you have
the feature (feature_link) and imdb (imdb_link) data links.

```
cd ~/pythia
mkdir -p data && cd data
wget http://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz

# The following command should result in a 'vocabs' folder in your data dir
tar xf vocab.tar.gz

# Download detectron weights
wget http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf detectron_weights.tar.gz

# Now download the features required, feature link is taken from the table below
# These two commands below can take time
wget feature_link

# [features].tar.gz is the file you just downloaded, replace that with your file's name
tar xf [features].tar.gz

# Make imdb folder and download required imdb
mkdir -p imdb && cd imdb
wget imdb_link

# [imdb].tar.gz is the file you just downloaded, replace that with your file's name
tar xf [imdb].tar.gz
```

```
cd ~/pythia
python tools/run.py --datasets textvqa --model lorra --config configs/vqa/textvqa/lorra.yml 
```

**Note for BUTD model :**  for training BUTD model use the config `butd.yml`. Training uses greedy decoding for validation. Currently we do not have support to train the model using beam search decoding validation. We will add that support soon. For inference only use `butd_beam_search.yml` config that supports beam search decoding.

## Pretrained Models

We are including some of the pretrained models as described in the table above.
For e.g. to run the inference using LoRRA for TextVQA for EvalAI use following commands:


```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install -e .
```

## License

Pythia is licensed under BSD license available in [LICENSE](LICENSE) file
