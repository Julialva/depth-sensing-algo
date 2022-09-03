#!bin/bash

pwd #estou no root directory
cd left_batch
mkdir leftinho
cp `ls | head -500` ./leftinho/

cd ..
cd right_batch
mkdir rightinho
cp `ls | head -500` ./rightinho/