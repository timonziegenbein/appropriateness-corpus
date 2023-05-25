To run the multilabel_roberta_baseline.py you can specify the following command line arguments:

* `--input` the input directory of the dataset split into folds
* `--output` the output directory to store the models
* `--fold ` the number of the fold in the input file
* `--repeat` the number of the k-fold repetition in the input file
* `--issue` to include the issue
* `--shuffle` to shuffle the words in the input

Example:
```bash
python multilabel_deberta_baseline.py --input ../../data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv --output ../../data/models/multilabel-roberta-baseline --fold 0 --repeat 0 --issue --shuffle
```
