# Masters Project
Learning to select cloze questions for KB style question answering

The project file structure follows 
[this](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3)
general template.

Models can be trained or tested by running main.py with the appropriate command line options, described within the script.

There are four different model classes: 

BasicModel - defined in basic_model.py, which uses no generative regularisation (although some mixture of KBP and Cloze data may still be
used by varying the generative regularisation strength (lambda) parameter).

ReinforceModelBlackBox - defined in reinforce_model_black_box.py, which uses the cluster based sampler for generative regularisation with the Cloze data.

ReinforceModelBlackBoxReader - defined in reinforce_model_black_box_reader.py, which uses the reader sampler for generative regularisation with the Cloze data.

ReinforceModel - defined in reinforce_model.py, which is not fully implemented but aims to use the standard GeneRe algorithm with the reader sampler. 

\
In order to train these models the data must first be converted to .TFRecords format by running the build_tfrecords_dataset.py 
and build_categorised_tfrecords_dataset.py scripts. These assume the raw data is stored in the correct directory structure.
This must be .../data/raw/[cloze, kbp]/[train, dev, test]/ so as an example the raw kbp training data would be 
stored in .../data/raw/kbp/train/. 

\
Please contact me at ucabain@ucl.ac.uk if there are any issues running the models or if the data is required.