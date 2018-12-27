input_folder="./data/avery/GSE37199/"
input_path="${input_folder}clinical.csv"

output_folder="./data/output/scaling/"
mkdir -p $output_folder

source activate py36
python --version

for scaling_method in "linear" "sigmoid"
do
    python -m src.autoencoder $input_path \
        --output-file ${input_folder}${scaling_method}.csv \
        --layers 10 \
        --batch-col "plate" \
        --scaling $scaling_method

done
