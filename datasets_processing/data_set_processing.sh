
CHBMIT_PATH=../../datasets/CHB-MIT
SHHS_PATH='/srv/local/data/SHHS'

cd datasets_processing
# Process CHB-MIT dataset
# python './CHB-MIT/process_1.py' --signals_path $CHBMIT_PATH 
# python './CHB-MIT/process_2.py' --signals_path $CHBMIT_PATH
# clean segments will be saved at $CHBMIT_PATH/clean_segments


# Process SHHS
python './SHHS/shhs_process.py' --windowsize 30 --multiprocess 30 --signals_path  $SHHS_PATH

