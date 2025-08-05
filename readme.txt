Implementation for "iDeepPE"

1) preprocess.py : make JSON files for the dataset for beamforming, it makes tr.json, dt.json, and et.json files.

python preprocess.py --in-dir path_to_6ch_noisy_CHiMEDB --in-dir-ext path_to_6ch_clean_and_noise_CHiMEDB --out-dir path_to_output_directory

2) make_mvdr_out.py: Obtain oracle MVDR beamformed output, which will be used for train_PF

python make_mvdr_out.py --tt-json path_to_JSON_files(tr,dt) --out-dir output_directory --mode tr_or_dt 

3) preprocess_post.py:  make JSON files for the dataset for postfiltering. it makes oracle_MVDR_tr.json, oracle_MVDR_dt.json, and oracle_MVDR_et.json files.

python preprocess_post.py --in-dir path_to_oracle_MVDR_beamforming_out_directory --out-dir path_to_output_directory

4) train_BF.py : DNN training code for beamforming

python train_BF.py --tr-json path_to_tr.json --cv-json path_to_dt.json --batch-size num_batch --epochs num_epoch --save-folder path_to_model_save_directory

5) train_PF.py : DNN training code for postfiltering

python train_PF.py --tr-json path_to_oracle_MVDR_tr.json --cv-json path_to_oracle_MVDR_dt.json --batch-size num_batch --epochs num_epoch --save-folder path_to_model_save_directory

6) evaluate.py : evaluate DNN_based beamforming and postfilter for test dataset

python evaluate.py --modelBF-path path_to_model_BF --modelPF-path path_to_model_PF --tt-json path_to_et.json
