data_dir="./data/feature_selection/"
output_dir="./data/feature_selection_output/"

source activate py36
python --version

for path in $(ls $data_dir)
do
    for column in 'centre' 'plate'
    do
        python -m src.autoencoder ${data_dir}${path} -o ${output_dir}${path} -l 10 -b $column
    done
done
