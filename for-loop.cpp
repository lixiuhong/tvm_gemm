for (int m = 0; m < M / factor_m; ++m) {    //多少个block_x
  for (int n = 0; n < N / factor_n; ++n) {  //多少个block_y

    //在(block_x, block_y)这样的分块内部
    for (int bm = 0; bm < factor_m; ++bm) {    //多少个thread_x
      for (int bn = 0; bn < factor_n; ++bn) {  //多少个thread_y

        //在这里会allocate local memory
        //在这里会allocate shared memory

        for (int vtm = 0; vtm < vthread_m; ++vtm) {  //这一维度会做unroll
          for (int vtn = 0; vtn < vthread_n; ++vtn) {  //这一维度会做unroll
            for (int tm = 0; tm < factor_m / vthread_m;
                 ++tm) {  //这一维度会保留
              for (int tn = 0; tn < factor_n / vthread_n;
                   ++tn) {  //这一维度会保留
              }
            }
          }
        }
      }
    }
  }
}

//最终vtm和vtn两个维度会在最内部展开