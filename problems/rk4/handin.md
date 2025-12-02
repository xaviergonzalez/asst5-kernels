# Optimizing RK4 kernel

xavier18

## Part 1: Steps Taken

### Step 0: Getting to the initial (correct) implementation

Even getting to the correct initial implementation was difficult for me. First, because we were dealing with 3D tensors, I didn't want to do rasterizing logic by hand and get errors. Therefore, I used the `packed_accessor32` feature of PyTorch's cpp implementation to allow for more natural indexing.

Also, I knew there were going to be a lot of reused parameters (all the shared paramters of heat diffusion and RK4), so following PA3 I set them up as `__constant__` memories to allow for faster memory access of these repeated parameters.

Finally, I really wanted to find a way to fuse all the RK4 steps together. However, I didn't find a way to get around the sequential dependencies of `k2` depending on `k1`, `k3` depending on `k2`, and so on. I would have liked to use `__syncthreads`, but I know that `__syncthreads` only works within a block. However, because the dimension of our grid is `600 x 600 x 600`, and we can only put a maximum of 1024 threads in a block, we can't fit the entire grid into a single block (if I'm using a single thread per single output pixel). What I'm afraid of is one thread getting ahead, and drawing from previous k1 values that haven't been updated to compute it's k2. In principle this could be done if each thread block took care of it's own region, but I wasn't sure how to deal with boundary conditions between regions. 

So, as a first stab, I implemented each RK4 step as a separate kernel launch. This was very straightforward, and I was able to get a correct implementation this way.

### Step 1: Profiling the initial implementation

> How is the code structured in the current step?



> What is the performance of the code (runtime)
> What other statistics did you measure and look at for the current code?
> What did you conclude from the measurements?
> What was your hypothesis about what was limited performance? (Or how it might be improved?) How did you come to this hypothesis?
> What does the hypothesis suggest you should try next in terms of how to modify your code's design?

### Step 2: Reducing recomputation

> How is the code structured in the current step? 
> What is the performance of the code (runtime)
> What other statistics did you measure and look at for the current code?
> What did you conclude from the measurements?
> What was your hypothesis about what was limited performance? (Or how it might be improved?) How did you come to this hypothesis?
> What does the hypothesis suggest you should try next in terms of how to modify your code's design?



## Part 2: Stopping

## Part 3: LLMs