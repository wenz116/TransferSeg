# $1 GPU_ID
# $2 VIDEO_NAME

cp ../nets/solver_template.prototxt ../nets/solver_$2.prototxt
cp ../nets/train_template.prototxt ../nets/train_$2.prototxt
cp ../nets/val_template.prototxt ../nets/val_$2.prototxt

sed -i 's/${VIDEO_NAME}/'$2'/g' ../nets/solver_$2.prototxt
sed -i 's/${VIDEO_NAME}/'$2'/g' ../nets/train_$2.prototxt
sed -i 's/${VIDEO_NAME}/'$2'/g' ../nets/val_$2.prototxt

mkdir -p ../output
mkdir -p ../output/figure
mkdir -p ../output/result
mkdir -p ../output/snapshot

python solve.py $1 $2