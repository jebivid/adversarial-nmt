# adversarial char-nmt

This is the code for COLING'18 paper: On Adversarial Examples for Character-Level Neural Machine Translation. It is largely based on [Yoon Kim's seq2seq implementation](https://github.com/harvardnlp/seq2seq-attn), and has similar installation requirements. You also need to use their processing script before training. This codebase deos NOT supprt bidirectional lstm, shards for large training data, and alignments, which are supprted in the original implementation. 

### Prerequisites

cunn and cutorch are required.
 
### Running

To perform training, use: 

```
th adversarial_train.lua  -data_file PathToTrainSET -val_data_file PathToValSET 
```
You need to set the flag -MT_type to either white of vanilla to perform adversarial or vanilla training, and set the flag -language cs/de/fr for different distributions of adversarial manipulations.

To perform controlled/targeted attacks on the saved models, use

```
th attack.lua  -test_data_file PathToTestSET -saved_model model.t7  -controlled 1 
```
You need to set the flags -targ_dict, -char-dict, and src-dict. For targeted attacks use -controlled 0.

## License

This project is licensed under the MIT License.
