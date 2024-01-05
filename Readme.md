# An implementation of [LMW23](https://eprint.iacr.org/2022/1703)-style doubly efficient PIR

This is the code accompanying our paper [OPPW23](https://eprint.iacr.org/2023/1510.pdf).
It is written in Rust, based only on the feanor-math library.
Currently, the asynchronous reading from an SSD is only implemented on Windows. 

# Computing the preevaluation databases

In addition to the main implementation in Rust, we accelerate the computation of the evaluation database using GPUs.
This is the code in `poly-batch-eval`.
When running it, make sure that the parameters `d` and `m` match the ones of the interpolation polynomial. 
The implementation is quite naive, since the performance of this step is not the focus of our work.