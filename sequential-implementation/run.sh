mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

./lbfgs_example