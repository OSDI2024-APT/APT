
mkdir build
cd ./build
cmake -DCMAKE_BUILD_TYPE=Release  -DCAFFE2_USE_CUDNN=1 .. && make -j 16
cd ../python 
python setup.py install
cd ..