The directory slt_arctic_demo_data contains demo data for the version of wavenet that uses local conditioning. I decided to include only these 60 utterances because I do not have explicit permission to include the whole arctic database in this fork. Instructions on how to download and use the whole database are in the last paragraph of this README file. 

Now, I will describe how to test the wavenet program with the demo data. 
In directory slt_arctic_demo_data there are two directories the binary_label_425 and the wav. The wav files are sampled at 16000 samples per second. The binary labels have been created from the state aligned labels and are time aligned to the corresponding wav files. Each binary labels file contains a matrix of number_of_frames x 425 float32 (in python format) (and 425 x number_of_frames float32 in Matlab format), where each new frame (row) is considered every 0.005 sec. Therefore each row of a binary label file represents 16000*0.005 = 80 samples from the corresponding wav file. Each row of a binary label file has 416 answers to questions from the question file questions-radio_dnn_416.hed plus 9 additional entries which correspond to the state (2,3,4,5,6) of the frame, the duration of the state that the frame belongs, the position of the frame within the state and within the phoneme etc. We may choose to inlude these 9 entries or to leave them out. Some of the 425 entries are not 0-1 normalized. Therefore, we should normalize them, by running the python script nomalize_labels.py. This script has also an option to include or not the 9 additional enties.
The modified files are saved in directory binary_label_norm.
Edit the nomalize_labels.py to fix the directories of the files and to choose either to min-max normalize the labels or to exclude entries 417:425.

Note the the total duration (in frames) of the binary labels and the wav files may differ by a few frames. These are usually silence frames at the end of the utterance which are omitted by the label generation file. This small inconsistency is handled by the reader of wavenet. 

     
In order to use the full slt_arctic database download Merlin from https://github.com/CSTR-Edinburgh/merlin. 
Then edit the script merlin/egs/slt_arctic/s1/run_full_voice.sh and put comments on steps 4 and 5. Then run the script (only steps 1, 2, 3).
It is important not to run the step 4, because this step destroys the time alignment between the labels and the wav files. 
At the beginning of step 3 the files in directory merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/acoustic_model/data/binary_label_425 are created. Copy the whole directory binary_label_425 and paste it in the Wavenet data directory. Also copy the directory wav (merlin/egs/slt_arctic/s1/slt_arctic_full_data/wav) and copy it in the wavenet data directory. Then run the nomalize_labels.py python script.


