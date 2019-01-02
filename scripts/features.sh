data_dir="./data/feature_selection/"
output_dir="./data/output/features/"

mkdir -p $output_dir

log_folder="./data/metrics/"
mkdir -p $log_folder

log_file="${log_folder}features.csv"

source activate py36
python --version

for path in $(ls $data_dir)
do
    for column in 'centre' 'plate'
    do
        python -m src.autoencoder ${data_dir}${path} -o ${output_dir}${path} -l 10 -b $column \
            --log-file $log_file
    done
done
