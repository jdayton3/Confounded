input_folder="./data/avery/GSE37199/"
input_path="${input_folder}clinical.csv"

output_folder="./data/output/loss_weight/"
mkdir -p $output_folder

source activate py36
python --version

for weight in 0.01 0.05 0.1 0.25 0.5 1.0 2.0 4.0 10.0 20.0 100.0
do
    printf -v weight_padded "%03.2f" $weight
    python -m src.autoencoder $input_path \
        --output-file ${output_folder}disc_weight_${weight}.csv \
        --layers 10 \
        --batch-col "plate" \
        --loss-weight $weight
done
