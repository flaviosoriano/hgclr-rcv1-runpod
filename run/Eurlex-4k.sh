data_name=Eurlex-4k
model_name=HGCLR
fold_idx=$1

# preprocess
# TODO: Preprocess read the samples.pkl and the split files (train.pkl,val.pkl,test.pkl) based on fold_idx (each split file contains a list of sample idx to filter samples from samples.pkl).
time_start=$(date '+%Y-%m-%d %H:%M:%S')
python main.py task=[preprecess] data_name=$data_name fold_idx=$fold_idx ...
        2. split the samples in 3 set: train.json, val.json, test.json and
        3. use baseline preprocess function to generate the required files to train, predict and eval
        4. store the preprocessed train, val test files in the dataset/$data_name/
        5.
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > resource/time/preprocess_${data}_${fold_idx}.tmr

# training
5. use the baseline script to train.py the model (using the specified hparams) and save the model chekpoint in /resource/model_checkpoint/${model_name}_${data_name}_${fold_idx}.<extention used by the baseline>


# predicting
6. use the chekpoint to predict (use the test.py script)


# eval
7. eval