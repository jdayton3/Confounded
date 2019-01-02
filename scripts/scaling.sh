input_folder="./data/avery/GSE37199/"
input_path="${input_folder}clinical.csv"

output_folder="./data/output/scaling/"
mkdir -p $output_folder

log_folder="./data/metrics/"
mkdir -p $log_folder

log_file="${log_folder}scaling.csv"

source activate py36
python --version

for scaling_method in "linear" "sigmoid"
do
    python -m src.autoencoder $input_path \
        --output-file ${output_folder}${scaling_method}.csv \
        --layers 10 \
        --batch-col "plate" \
        --scaling $scaling_method \
        --log-file $log_file

done
