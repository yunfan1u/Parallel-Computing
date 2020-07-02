# HW3: Sobel

## Goal:
* Parallelize a 5x5 variant of the sobel operator with CUDA.

## Implementation:
將sequential code中的sobel()函式改成CUDA版本：

### 1. Computation
*    原本是height, width的的兩層迴圈，改成用GPU平行做。
    變數x, y定義為：
```
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```
* 計算部分跟sequentual一樣，但是需要有
```
if((x>=0 && x < width) && (y>=0 && y<height))
```
條件式確保x, y不超出範圍。


### 2. Communication
**CUDA memory model**為CPU(host)和GPU(device)各自有自己的memory system，兩者需要communication傳遞才能使用資料。
* 宣告4個buffer: 
        1. `src_img`: 讀入的圖片。
        2. `src_cuda`: 使用cudaMemcpy()將src_img傳到device端。
        3. `dst_img`: 輸出的圖片。
        4. `dst_cuda`: 在device端的計算結果，使用cudaMemcpy()將dst_cuda傳到host端。


另外，要在GPU使用的變數必須加上__device__
```
__device__ int mask[MASK_N][MASK_X][MASK_Y]
```

## Performance Tuning
需要了解自己使用的GPU架構，才能有好的使用效能。
本次是使用**Nvidia GeForce GTX 1080**這台GPU。

wikipidia可以找到Compute capability:
https://en.wikipedia.org/wiki/CUDA

* GTX 1080的Compute capability為6.1
* Maximum number of threads per block = 1024

所以定義thread per block為32*32 = 1024。
```
dim3 threadsPerBlock(32, 32);
dim3 blocksPerGrid((width+31)/32, (height+31)/32);
```
最後，加上`cudaThreadSynchronize()`
CUDA版本的sobel就可以運作了。

```
sobel_gpu<<<blocksPerGrid, threadsPerBlock>>>(src_cuda, dst_cuda, height, width, channels);
cudaThreadSynchronize();
```

## Discussion
Nvidia 6引進了**Unified Memory**，即CPU和GPU可以共享managed memory。
讓programmer可以更容易使用CUDA，處理比較複雜的資料結構。

![](https://i.imgur.com/RyYYDjB.png)

**cudaMallocManaged vs. cudaMalloc**
如果要使用**Unified Memory**，使用前者配置記憶體。

### cudaMalloc
* Allocate memory on the device.

### cudaMallocManaged
* Allocates memory that will be automatically managed by the **Unified Memory** system.

有benchmark顯示使用cudaMallocManaged效能會略低於cudaMalloc。
CPU與GPU之間的bottleneck是PCIE bus

## Result

Original image:
![](https://i.imgur.com/Gbkw8tq.png)

After Sobel filter:
![](https://i.imgur.com/MQenfAw.jpg)
