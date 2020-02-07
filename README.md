Times obtained from an average of 20 runs.
![plotz](https://github.com/DolicaAkelloEgwel/howdoespycudawork/blob/master/Figure_1.png "")  
  
Notes:  
- `cupy` seems to wipe the floor with ~`pycuda`~ _everything else_.   
- This isn't taking transferring into account.  
- Strangely `cupy` finds "background correction" easier than adding arrays.  
- It seems like `pycuda` has to carry out some complication (I am unsure how to isolate it) which may be part of the reason it appears slower.  
- `cupy` doesn't seem to be bothered when doing "background correction" for larger and larger arrays.  
- I suspect `cupy` will be much simpler to use in the long-run.
