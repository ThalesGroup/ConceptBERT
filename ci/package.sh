#/bin/bash
set -e

BASE_PATH=$(dirname "$0")/..

PACKAGE_VERSION=$(python $BASE_PATH/setup.py --version)
TAG_VERSION=${PACKAGE_VERSION//+/-}

echo "Cleaning last build"
rm -rf $BASE_PATH/dist
mkdir $BASE_PATH/dist

echo ${TAG_VERSION} > $BASE_PATH/dist/__version__

echo "Building .whl for version ${TAG_VERSION}"
python $BASE_PATH/setup.py clean --all bdist_wheel

WHEEL_PACKAGE=$(ls $BASE_PATH/dist)
printf "\n\e[1;32mPackaged ${WHEEL_PACKAGE} successfully.\e[0m\n\n"
