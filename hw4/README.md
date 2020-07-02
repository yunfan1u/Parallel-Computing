# HW4: Bitcoin Miner

## Goal:
### Find out a proper hash value for a given block header
* The hash value has to be less than the Target Difficulty
* The time you spent on finding the hash value has to be accelerated


## Implementation:
本次作業的重點是使用CUDA平行化找Nonce的程式部分。

將sequential code中：
```
for(block.nonce=0x00000000; block.nonce<=0xffffffff;++block.nonce)
```
for loop迴圈平行化
因為0xffffffff很大，所以將它切成好幾個iteration執行
```
for(unsigned int i=0; i<65536; i++)    // 65536
{
    findNonce<<<blocksPerGrid, threadsPerBlock>>>(target_hex_device, sha256_ctx_device, block_device, i);
    cudaDeviceSynchronize();
}
```
我使用：
* threadsPerBlock = 256
* blockPerGrid = 256

總共需要執行65536次iterations，即一次iteration平行執行65536個threads。

宣告一個`__device__ __managed__`的global variable
```
__device__ __managed__ bool isFound;
```
為一個**unified memory**的全域變數。
如果找到滿足終止條件的Nonce，將isFound的flag設true，即可break出迴圈。
但還是需要待此iteration全部的threads跑完，才能break出迴圈。

### Kernel function:
`block`, `sha256_ctx`, `target_hex`也是使用cudaMalloc()和cudaMemcpy()，以pointer的的方式傳給kernel function。

再kernel function中，block和sha256_ctx會產生race condition，所以複製local variable給它。

做`double_sha256()`前也需要確保不發生race condition。
```
double_sha256(&local_sha256, (unsigned char*)&local_block, sizeof(local_block));
__syncthreads();
```

最後在`little_endian_bit_comparison()`成立時，再將值還給block和sha256_ctx。
```
*block = local_block;
*sha256_ctx = local_sha256;
```
這裡必須注意必須是複製value而不是address
```
cudaMemcpy(&sha256_ctx, sha256_ctx_device, sizeof(SHA256), cudaMemcpyDeviceToHost);
cudaMemcpy(&block, block_device, sizeof(HashBlock), cudaMemcpyDeviceToHost);
```
## Results
![](https://i.imgur.com/iixjaiJ.png)




## Homework review:
非常感謝助教不厭其煩的讓我問問題，最後才能完成這份作業QQ
