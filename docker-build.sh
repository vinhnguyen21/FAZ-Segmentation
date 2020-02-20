CWD=$(pwd)
# mkdir "${CWD}/images"
# mkdir "${CWD}/images/raw"
# mkdir "${CWD}/images/predict"
CWD="${CWD}/images"

docker build -t asia.gcr.io/a2ds-235802/faz-segmentation:latest -f ./docker/Dockerfile .
docker run -p 2001:2000 -v ${CWD}:/images -d asia.gcr.io/a2ds-235802/faz-segmentation 