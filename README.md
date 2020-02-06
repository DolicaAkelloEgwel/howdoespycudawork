Times obtained from an average of 10 runs.
![plotz](https://github.com/DolicaAkelloEgwel/howdoespycudawork/blob/master/Figure_1.png "")  
  
Interesting features:  
- `cupy` seems to wipe the floor with `pycuda`.  
- This isn't taking transferring into account.  
- Strangely `cupy` finds "background correction" easier than adding arrays.  
- It seems like `pycuda` has to carry out some complication (I am unsure how to isolate it) which may be part of the reason it appears slower.  
- `cupy` doesn't seem to be bothered when doing "background correction" for larger and larger arrays.  
