make
if [ ! -e text8 ]; then
  #wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  #gzip -d text8.gz -f
  curl -o text8.zip http://mattmahoney.net/dc/text8.zip
  unzip text8.zip
fi
time ./term_word2vec -train text8 -output vectors.bin -cbow 0 -size 200 -window 5 -negative 5 -hs 0 -sample 1e-3 -threads 3
./distance vectors.bin
