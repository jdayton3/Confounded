input_path="./data/mnist/noisy.csv"
output_dir="./data/output/mnist/"
output_path="${output_dir}mnist.csv"

mkdir -p $output_dir

log_folder="./data/metrics/"
mkdir -p $log_folder

log_file="${log_folder}mnist.csv"

source activate py36
python --version


python -m src.autoencoder $input_path \
    --output-file $output_path \
    --layers 10 \
    --ae-layers 2 \
    --batch-col "Batch" \
    --code-size 200 \
    --scaling linear \
    --loss-weight 1 \
    --log-file $log_file

bash "../scripts/mnist.py"
