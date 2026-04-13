# dlib_train

### what is it?

scripts for [WpfMusicPlayer](https://github.com/WpfMusicPlayer-Dev/WpfMusicPlayer) to train model weights used in LRC detection.

specialized for CJK lyrics.

line_net can detect language type (7 types):

zh, en, jp, kr, jyut, roma, onomatopoeia

song_net can detected language classification (12 types):

- 1-lang only: zh_only, en_only, jp_only, kr_only
- 2-lang combination: en_zh_trans, jp_zh_trans, kr_zh_trans, jp_roma, kr_roma, zh_jyut
- 3-lang combination: jp_zh_trans_roma, kr_zh_trans_roma 

### how to build & train?

build: run `vcpkg integrate install` in your developer powershell, modify dlib_train.cpp to enable the function you need, then run.

train: put train.tsv (line_net train data) and song_train.tsv (song_net train data) in dlib_train/ and follow the specification of training data. must be in UTF-8 encoding.

lang_type and song_sequence_type are listed above.
#### train.tsv
[lang_type] [sentence]

#### song_train.tsv
[song_sequence_type] [lang_type sequence, like "zh jp zh jp zh jp" (do not include quotation marks in your dataset.)]

you can find the model weights in dlib_train/. line_net -> lyric_lang_mlp.dat, song_net -> song_structure_mlp.dat

### other information

Licensed under the MIT license.

Feel free to use in any project if it's useful. compatible with most C++-based project.

Depends: dlib (managed by vcpkg. if you need to use CUDA acceleration, build by yourself please.)