input_folder="./data/avery/GSE37199/"
input_path="${input_folder}clinical.csv"

output_folder="./data/output/layers/"
mkdir -p $output_folder

log_folder="./data/metrics/"
mkdir -p $log_folder

log_file="${log_folder}layers.csv"

source activate py36
python --version

for i in $(seq 0 20)
do
    printf -v padded "%02d" $i
    python -m src.autoencoder $input_path -o ${output_folder}layers_${padded}.csv -l $i \
        --log-file $log_file
done
