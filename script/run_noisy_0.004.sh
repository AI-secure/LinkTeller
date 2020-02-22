set -x


python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.004
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.008
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.012
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.016
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.020
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.024
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.028
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.032
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.036
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.400

python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.004
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.008
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.012
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.016
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.020
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.024
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.028
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.032
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.036
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.040

python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.004
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.008
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.012
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.016
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.020
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.024
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.028
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.032
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.036
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.040


python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.004
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.008
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.012
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.016
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.020
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.024
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.028
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.032
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.036
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.040

python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.004
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.008
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.012
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.016
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.020
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.024
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.028
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.032
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.036
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.040

python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.004
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.008
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.012
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.016
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.020
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.024
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.028
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.032
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.036
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.040

