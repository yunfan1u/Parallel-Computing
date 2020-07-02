# HW2: Mandelbrot Rendering

## Goal:
* Implement a Ray Marching algorithm to render the “Classic” Mandelbulb.
* TAs provide the sequential code, and our job is to parallelize it with MPI and openMP.

## Implementation:
主要架構是使用MPI process分配工作，每個process有各自的thread透過openMP平行。

### MPI
* Distributed memory
* 因為工作量是可以預期的，故使用Static scheduling。
* 工作分配策略是：
`if((i * width + j) % size == rank)`就分配到工作，最後再使用`MPI_Reduce`將local data加總起來。
* 這樣工作分配比起直接將圖片切成size部分，有更好的Load balance。
* 因為Mandelbrot set有對稱圖形的特性，所以Static工作分配可以得到還不錯的Load Balance。

### openMP
* Shared memory
* openMP的平行化是根據num_threads的數目去分配執行緒的工作，我使用的是`schedule(dynamic)`去分配rendering的兩層迴圈。
* 測試過`schedule(static)`的效果沒有dynamic來得好。

### Optimization
* 程式中互相獨立的部分使用使用`#pragma omp parallel sections`做平行化。
* 其他functions中的迴圈有嘗試使用openMP平行，但是效能並沒有提升，所以還是保留原版本。


## Discussion:
* 平行程式本身就是一個資源分配的課題，有效分配工作量給各個workers會提升平行運算的效能。
*  本次作業的重點應該是MPI的scheduling問題，因為時間的關係沒有寫dynamic版本，之後應該要完成兩個版本的比較。

## Homework review:
* 學到了static/dynamic scheduling的概念。
* Load balance的重要。




