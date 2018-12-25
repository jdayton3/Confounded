input_folder="./data/avery/GSE37199/"
input_path="${input_folder}tidy.csv"

source activate py36
python --version

for i in $(seq 11 20)
do
    printf -v padded "%02d" $i
    python -m src.autoencoder $input_path -o ${input_folder}layers_${padded}.csv -l $i
done

