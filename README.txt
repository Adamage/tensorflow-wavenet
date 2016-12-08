This is an experimental version which is released so that the community to test an idea of how local conditioning can be implemented in wavenets.
The program was created for personal use and is not compatible with the main version of wavenets. 
It uses full context labels, which hopefully are time aligned to wav files. The wav files and the corresponding labels can be found in one
of the examples of Merlin (see the README.txt in data directory).
The program reads files using a list of train filenames and a list of test filenames which ara in ./data/ directory.
There are several open issues, especially on the synthesis part of the program. The program may also contain several bugs. The flag 
--fast_generation=false has not been tested and it will probably has out of index problems at the end of the utterance.

Change the data directory in train.py (DATA_DIRECTORY = './data/slt_arctic_demo_data/')
You may need to change LABEL_DIM = 425 
run: python train.py

