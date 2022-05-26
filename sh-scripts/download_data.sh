#!/bin/bash

data_dir=./data
train_squad_data_name=train-v1.1.json
valid_squad_data_name=dev-v1.1.json
pretrained_glove_data_name=glove.840B.300d.zip

if [ ! -e $data_dir ];
then
  mkdir $data_dir
  echo "==========>>>>>>>>>> Data directory [$data_dir] created!"
else
  echo "==========>>>>>>>>>> Data directory [$data_dir] already exists!!"
fi

train_data_file=$data_dir/$train_squad_data_name
if [ ! -e $train_data_file ];
then
  wget --no-check-certificate \
    https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json \
    -O $data_dir/$train_squad_data_name
  echo "==========>>>>>>>>>> Train data downloaded at [$train_data_file]"
else
  echo "==========>>>>>>>>>> Train data [$train_data_file] already exists!!"
fi

valid_data_file=$data_dir/$valid_squad_data_name
if [ ! -e $valid_data_file ];
then
  wget --no-check-certificate \
    https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json \
    -O $valid_data_file
  echo "==========>>>>>>>>>> Validation data downloaded at [$valid_data_file]"
else
  echo "==========>>>>>>>>>> Train data [$valid_data_file] already exists!!"
fi

pretrained_glove_file=$data_dir/$pretrained_glove_data_name
if [ ! -e $pretrained_glove_file ];
then
  wget --no-check-certificate \
    http://nlp.stanford.edu/data/glove.840B.300d.zip \
    -O $pretrained_glove_file
  echo "==========>>>>>>>>>> Pretrained GloVe embeddings downloaded at [$pretrained_glove_file]"
else
  echo "==========>>>>>>>>>> Pretrained GloVe embeddings [$pretrained_glove_file] already exists!!"
fi

  unzip -q $pretrained_glove_file -d $data_dir
  # rm -r $pretrained_glove_file
  echo "==========>>>>>>>>>> Extracted pretrained GloVe embeddings from its zip archive."
