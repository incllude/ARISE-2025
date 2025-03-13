%%writefile train.sh
# export PYTHONPATH=$PYTHONPATH:/kaggle/working
# export HYDRA_FULL_ERROR=1
# echo "Updated PYTHONPATH: $PYTHONPATH"
accelerate launch classification/train.py