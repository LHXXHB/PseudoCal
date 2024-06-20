# Extract features of both source train/val and target data, which are required in CPCS and TransCal but not in our PseudoCal.
python feat_extract.py --pl bnm --s 0 --t 1 --gpu_id 0 --output logs/uda/train/ --net resnet50 --dset office-home --seed 22
python feat_extract.py --method PADA --s 0 --t 1 --gpu_id 0 --output logs/pda/train/ --net resnet50 --dset office-home --da pda --seed 22

# Calibrate the UDA model for unlabeled target data using different calibration methods.
python calib.py --pl bnm --s 0 --t 1 --gpu_id 0 --output_tr logs/uda/train/ --net resnet50 --dset office-home --seed 22
python calib.py --method PADA --s 0 --t 1 --gpu_id 0 --output_tr logs/pda/train/ --net resnet50 --dset office-home --da pda --seed 22