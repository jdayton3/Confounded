input_folder="./data/avery/GSE37199/"
input_path="${input_folder}clinical.csv"

output_folder="./data/output/ae_layers/"
mkdir -p $output_folder

source activate py36
python --version

for layer in 0 1 2 3
do
    python -m src.autoencoder $input_path \
        --output-file ${output_folder}ae_layers_${layer}.csv \
        --layers 10 \
        --batch-col "plate" \
        --ae-layers $layer
done
