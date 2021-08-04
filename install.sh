#!/bin/bash

# # Read command line argument
VERSION=0.6.1

echo "AI4PROD VERSION ": $VERSION
# get current user not sudo
[ $SUDO_USER ] && user=$SUDO_USER || user=$(whoami)

git clone https://github.com/ko1nksm/getoptions.git
cd getoptions
make
make install
cd ..
rm -r getoptions

parser_definition() {
    setup REST help:usage -- "Usage: install.sh [options]... [arguments]..." ''
    msg -- 'Options:'
    #flag withut param flag FLAG -f --flag -- "takes no arguments"
    flag CMAKE --cmake -- "takes no arguments"
    flag CPU --cpu -- "takes no arguments"
    param CUDA --cuda -- "takes one argument"
    disp :usage -h --help
    disp VERSION --version
}

# ubuntu deps
apt-get install wget

eval "$(getoptions parser_definition) exit 1"

if [ $CMAKE ]; then
    echo "INSTALL CMAKE "
    sudo -u $user wget http://www.cmake.org/files/v3.14/cmake-3.14.4.tar.gz
    sudo -u $user tar -xvzf cmake-3.14.4.tar.gz
    cd cmake-3.14.4
    apt-get install -y libcurl4-gnutls-dev zlib1g-dev
    ./bootstrap --system-curl
    make -j10
    make install
    cd ..
    rm cmake-3.14.4.tar.gz
    rm -r cmake-3.14.4
fi



# Check if binaries are present for correct cuda
# need to add spaces before and after [ value ]
if [ $CUDA = 10.2 ]; then
    echo "VALID CUDA"
    # To assign a value is not possibile to have spaces around =
    DEPS="1QmJSFbM1KW4nih5Wy2Ft4pagJOTipaZ6"
elif [ $CUDA = 11.1 ]; then
    DEPS="1rB2mnisEsR2oy8sishyAaTI6O2X-36Lh"
elif [ $CPU ]; then
    echo "CPU"
    DEPS="1w7TE6QmZ9LxVze7652N7hITF4gdxdAsQ"
else

    echo "DEPENDENCIES VERSION NOT FOUND WITH THIS CONFIGURATION "
    exit
fi



echo "CUDNN: $CUDNN, CUDA: $CUDA, OPTION: $OPTION"

echo "DEPS": $DEPS

sudo -u $user mkdir deps

cd deps

sudo -u $user wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$DEPS" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$DEPS" -O deps.tar.xz && rm -rf /tmp/cookies.txt

echo "UNZIP DEPS. COULD TAKE A WHILE"

sudo -u $user tar -xf deps.tar.xz
rm deps.tar.xz

# move tensorrt folder to /usr/local

mv tensorrt /usr/local

# return to HOME
cd ..


# install vcpkg
sudo -u $user git clone https://github.com/microsoft/vcpkg.git

cd vcpkg
git checkout e9ff3cd5a04cd0e8122ff56e9873985ff71aa3ca
./bootstrap-vcpkg.sh
./vcpkg install yaml-cpp
./vcpkg install jsoncpp
./vcpkg install catch2

# add correct permission to vcpkg
cd ..
chmod -R 777 vcpkg
echo "AI4PROD DEPS INSTALLED SUCCESSFULLY NOW COMPILE WITH CMAKE FOLLOWING DOCUMENTATION"

#chmod -R 777 vcpkg

# COMPILE
# mkdir build
# cd build

# echo $user
# sudo -u $user mkdir build

# cmake -DCMAKE_INSTALL_PREFIX=$INSTALL -DEXECUTION_PROVIDER=$PROVIDER -DADDTEST=$DDTEST ..

# make install
