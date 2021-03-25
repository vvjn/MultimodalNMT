# PREPROCESS TEXT
python ../MultimodalNMT/preprocess.py -train_src ../multi30k-wmt16/en-de/train.norm.tok.lc.bpe10000.en -train_tgt ../multi30k-wmt16/en-de/train.norm.tok.lc.bpe10000.de -valid_src ../multi30k-wmt16/en-de/val.norm.tok.lc.bpe10000.en -valid_tgt ../multi30k-wmt16/en-de/val.norm.tok.lc.bpe10000.de -save_data data/m30k

# PREPROCESS IMAGES
python ../MultimodalNMT/extract_image_features.py --gpuid 0 --pretrained_cnn vgg19_bn --splits=train,valid,test --images_path ../multi30k-images/flickr30k-images/ --train_fnames ../multi30k-wmt16/raw/train_images.txt --valid_fnames ../multi30k-wmt16/raw/val_images.txt --test_fnames ../multi30k-wmt16/raw/test_images.txt

# TRAIN IMGD MODEL
python ../MultimodalNMT/train_mm.py -data data/m30k -save_model model_snapshots/IMGD_ADAM -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --multimodal_model_type imgd

# TRAIN IMG+SRC MODEL
python ../MultimodalNMT/train_mm.py -data data/m30k -save_model model_snapshots/IMG_SRC_ADAM -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --multimodal_model_type src+img -decoder_type doubly-attentive-rnn

# TRANSLATE
python ../MultimodalNMT/translate_mm.py -gpuid 0 -src ../multi30k-wmt16/en-de/test.norm.tok.lc.bpe10000.en -model model_snapshots/${SELECT_MODEL} -path_to_test_img_feats ./flickr30k_test_vgg19_bn_cnn_features.hdf5 -output model_snapshots_results/${SELECT_MODEL}_test

# CALCULATE BLEU SCORE
perl ../MultimodalNMT/tools/multi-bleu-detok.perl ../multi30k-wmt16/en-de/test.norm.tok.lc.de < model_snapshots_results/${SELECT_MODEL}_test

