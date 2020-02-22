set -x


python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.01
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.02
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.03
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.04
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.05
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.06
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.07
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.08
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.09
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode vanilla --train-ratio 0.10

python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.01
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.02
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.03
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.04
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.05
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.06
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.07
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.08
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.09
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --break-down --train-ratio 0.10

python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.01
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.02
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.03
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.04
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.05
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.06
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.07
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.08
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.09
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method kmeans --n-clusters $4 --train-ratio 0.10


python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.01
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.02
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.03
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.04
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.05
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.06
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.07
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.08
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.09
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 42 --train-ratio 0.10

python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.01
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.02
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.03
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.04
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.05
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.06
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.07
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.08
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.09
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 51 --train-ratio 0.10

python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.01
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.02
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.03
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.04
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.05
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.06
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.07
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.08
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.09
python main.py --eps $1 --noise-seed $2 --dataset $3 --mode clusteradj --cluster-method random --n-clusters $4 --cluster-seed 92 --train-ratio 0.10

