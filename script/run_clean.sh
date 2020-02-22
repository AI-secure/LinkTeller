set -x


python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.01
python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.02
python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.03
python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.04
python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.05
python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.06
python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.07
python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.08
python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.09
python main.py --dataset $1 --mode vanilla-clean --train-ratio 0.10

python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.01
python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.02
python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.03
python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.04
python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.05
python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.06
python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.07
python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.08
python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.09
python main.py --dataset $1 --mode clusteradj-clean --break-down --train-ratio 0.10

python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.01
python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.02
python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.03
python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.04
python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.05
python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.06
python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.07
python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.08
python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.09
python main.py --dataset $1 --mode clusteradj-clean --cluster-method kmeans --n-clusters $2 --train-ratio 0.10


python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.01
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.02
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.03
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.04
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.05
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.06
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.07
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.08
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.09
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 42 --train-ratio 0.10

python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.01
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.02
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.03
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.04
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.05
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.06
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.07
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.08
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.09
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 51 --train-ratio 0.10

python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.01
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.02
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.03
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.04
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.05
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.06
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.07
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.08
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.09
python main.py --dataset $1 --mode clusteradj-clean --cluster-method random --n-clusters $2 --cluster-seed 92 --train-ratio 0.10

