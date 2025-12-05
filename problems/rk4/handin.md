# Optimizing RK4 kernel

xavier18

leaderboard name: ungulates

## Part 1: Steps Taken

### Step 0: Getting to the initial (correct) implementation

Even getting to the correct initial implementation was difficult for me. First, because we were dealing with 3D tensors, I didn't want to do rasterizing logic by hand and get errors. Therefore, I used the `packed_accessor32` feature of PyTorch's cpp implementation to allow for more natural indexing.

Also, I knew there were going to be a lot of reused parameters (all the shared paramters of heat diffusion and RK4), so following PA3 I set them up as `__constant__` memories to allow for faster memory access of these repeated parameters.

Finally, I really wanted to find a way to fuse all the RK4 steps together. However, I didn't find a way to get around the sequential dependencies of `k2` depending on `k1`, `k3` depending on `k2`, and so on. I would have liked to use `__syncthreads`, but I know that `__syncthreads` only works within a block. However, because the dimension of our grid is `600 x 600 x 600`, and we can only put a maximum of 1024 threads in a block, we can't fit the entire grid into a single block (if I'm using a single thread per single output pixel). What I'm afraid of is one thread getting ahead, and drawing from previous k1 values that haven't been updated to compute it's k2. In principle this could be done if each thread block took care of it's own region, but I wasn't sure how to deal with boundary conditions between regions. 

So, as a first stab, I implemented each RK4 step as a separate kernel launch. This was very straightforward, and I was able to get a correct implementation this way.

### Step 1: Profiling the initial implementation

> How is the code structured in the current step?

I store parameters of the stencil and rk4 in constant memory. After initializing I do memory io. I then set up torch tensors for `u` (the value of the potential at each point) as well as for each of the 4 steps in rk4. I then have a four loop for the steps of rk4, where at each step I do a separate kernel launch for each of the 4 k values and a final kernel to take the rk4 step. Each kernel naively computes the math formula as laid out in the instructions.

> What is the performance of the code (runtime)

184 ms

> What other statistics did you measure and look at for the current code?

I ran the profiler, which showed the the time taken by the first 20 kernel launches. It showed that my kernels for the laters steps (k2-k4) were around 2x as slow from the first kernel launch (k1) [4.4 ms vs 2.4 ms]

> What did you conclude from the measurements?

I concluded that a significant bottleneck in the code was occuring the k2-k4 kernels.

> What was your hypothesis about what was limited performance? (Or how it might be improved?) How did you come to this hypothesis?

Looking at math, I realized that I was recomputing previously computed information multiple times. For example, in k2, I was recomputing the stencil on `u` again, even though I had already computed it for k1. Similarly for k3 and k4, I was recomputing the stencil on `u` multiple times. This was leading to a lot of redundant computation.

> What does the hypothesis suggest you should try next in terms of how to modify your code's design?

From this profiling, I realized that I should redo the computation of the later steps (k2-k4) to reuse previously computed stencil values, rather than recomputing them from scratch.

### Step 2: Reducing recomputation

> How is the code structured in the current step? 

I rewrote the code so that the `k2-k4` kernels took in `k1` so as to avoid recomputation, reducing recomputation by half.

> What is the performance of the code (runtime)

126 ms

> What other statistics did you measure and look at for the current code?
> What did you conclude from the measurements?
> What was your hypothesis about what was limited performance? (Or how it might be improved?) How did you come to this hypothesis?
> What does the hypothesis suggest you should try next in terms of how to modify your code's design?

From the leaderboard, I see it's possible to do close to 2x better than my implementation. 

An obvious thing I could try would be to tune the block size. Right now I'm using a block size of `8 x 8 x 8 = 512` threads per block, which seems reasonable, but maybe I could do better.

Also, my implementation uses a lot of memory io, which is often a bottleneck. I'm not really traversing through the 3D space in a spatially aware way (I just compute `k1` across all space, then `k2`, etc.) Something I could do much better would be to be spatially aware, and instead load in 2D tiles, thus keeping many of the neighbors needed by the stencil in cache to reduce memory i/o. 

## Part 2: Stopping

Fundamentally, I stopped because I had beaten the naive cuda implementation, and was running out of time because I am taking pass/fail and presenting at neurips.

## Part 3: LLMs

I absolutely found the LLM assistant to be helpful.

I used LLMs at basically every stage of implementation, from helping me set up the initial cuda kernel, to helping me with debugging, to helping me with optimization ideas.

The LLM was least helpful for initial writing of the kernel. I was having to really review/relearn/teach myself a lot of cuda. At a couple of points I tried to ask LLMs to write my initial kernel first draft, but the code never seemed clean, correct, or right. So, most of the initial first draft was my own work. However, in this phase it was useful for targetted questions. For example, I knew that I didn't want to flatten my image an nd do rasterizing logic by hand, so I asked LLMs how to get more natural indexing. It suggested `packed_accessor32`, which worked well.

I had made a lot of syntax errors in my first run through (not indexing into the packed accessors correctly, using python syntax for the for loop), and LLMs were very helpful in debugging these syntax errors.

Finally, LLMs were very helpful in the optimization phase. I was running out of time and so simply copy pasted my first implementation and the profiling results, and it immediately suggested the optimization I used (avoid recomputation). This was enough to beat the naive cuda baseline.