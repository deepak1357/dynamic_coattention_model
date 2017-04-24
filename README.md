This is a tensorflow implementation of dynamic coattention network for reading comprehension.

https://arxiv.org/abs/1611.01604

tensorflow version r1.0

python 2.7

Pretrained glove vectors are required

http://nlp.stanford.edu/data/glove.840B.300d.zip

Data can be downloaded: 
https://rajpurkar.github.io/SQuAD-explorer/

put train-v1.1.json and dev-v1.1.json in ./data

run preprocess: python data_preprocess.py /path/to/glove.txt

Set parameter in config.py, 

to train the model: python DCN.py
