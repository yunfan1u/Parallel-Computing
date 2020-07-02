# HW5: N-body Simulation

## Goal:
Using CUDA to perform N-body simulation with 2 GPUs.

## Implementation:
### run_step() function
本次作業我將run_step()這個function改為GPU版本：
```
for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        ...
```
將n個星體平行去做計算，即
```
int i = blockDim.x * blockIdx.x + threadIdx.x;
if(i < n){
    for (int j = 0; j < n; j++)
    ...
}
```
將O(n^2)縮短為O(n)。
結束一個step後，需要將qx, qy, qz的變數傳會給Host端。


### Problem 3
Problem 3使用簡單的三角函數投影技巧去計算飛彈軌跡：
* 模擬每個device被破壞後是否可以避免planet被攻擊
* 被破壞的device質量歸0
* 破壞device後需將m傳回給Device端更新
* 持續進行step模擬直到結束


### CUDA
因為顯然Problem 3的執行時間會大於Problem 1和2，所以我將將Problem 1和2一起做，給GPU 0執行，Problem 3則給GPU 1執行。

run_step()需要傳入的變數有qx, qy, qz, vx, vy, vz, m, type。
宣告三份變數給三個問題。
其中Problem 1和2分別分配給stream1, 2進行**Concurrent Kernel Execution**

設定：
`threadsPerBlock = 32`
`blocksPerGrid = (n + threadsPerBlock-1)/threadsPerBlock`
其中n為星體數量。
有比較好的表現。

即
```
// Problem 1, 2
cudaSetDevice(0);
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

cudaMemcpyAsync(..., stream1)
cudaMemcpyAsync(..., stream2)
...

for (int step = 0; step <= param::n_steps; step++) {
    run_step_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>();  // for P1
    run_step_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>();  // for P2

    // other works
}

// Problem 3
cudaSetDevice(1);

cudaMalloc();

for(int i=0; i<devList.size(); i++){
            
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    cudaMemcpy();
    ...
    
    for (int step = 0; step <= param::n_steps; step++){
        run_step_kernel<<<blocksPerGrid, threadsPerBlock,>>>();
        
        // other works
    }
}


```

## Discussion:
### If there are 4 GPUs instead of 2, what would you do to maximize the performance?
可能會將2個GPU給Problem 1和2，剩下兩個再依device數量均分給Problem 3。


### If there are 5 gravity devices, is it necessary to simulate 5 n-body simulations independently for this problem?
如果將變數copy五份並且分別給不同的stream去執行，或許可以加速，以空間換取時間。


## Homework Review:
這次作業不算是很成功，因為只有達到correctness，沒有達到加速的部分。
可能是因為太頻繁communication，應該對題目個別去做GPU加速，而不是寫一個general的function。也應該要去做更大量的平行。
在小測資會跑的比sequential慢，比較大的則持平或略快。
最後在時間限制內只通過6/12個測資。
