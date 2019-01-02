input_folder="./data/avery/GSE37199/"
output_folder="./data/output/code_size/"
input_path="${input_folder}clinical.csv"

mkdir -p $output_folder

log_folder="./data/metrics/"
mkdir -p $log_folder

log_file="${log_folder}code_size.csv"

source activate py36
python --version

for i in 10 50 100 500 1000 5000 10000 20000
do
    printf -v padded "%05d" $i
    python -m src.autoencoder $input_path \
        --output-file ${output_folder}size_${padded}_plate.csv \
        --code-size $i \
        --batch-col "plate" \
        --log-file $log_file
done
