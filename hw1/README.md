# HW1: Sokoban

## Goal:
Implement a parallel program to solve the sokoban game.
    
## Pthread vs openMp?
### Pthread
- Fork-Join model
- Thread-based
- 自由度較高

### openMP
- Compiler directive based model
- Task-based
- 自由度較低，但操作上更高階層面，不用處理複雜的mutex問題，compiler會幫你完成。
- High portability

本次作業選用openMP來實作平行程式，因為比較方便外，也比較專注於平行程式的撰寫。

## Implementation:
* 以BFS的方式WDSA方向四面搜尋，嘗試Push或Move動作。並以`std::set`紀錄當前的map狀態，如果此步已在set就不會進行。
    過程中如果找到可行的solution，即return。

## Parallization:
* 在四個方向搜尋的時候有for loop可以做平行化。
    需注意的是openMP平行程式需要有良好的程式結構，即迴圈中不可以有break/return等動作。不過openMP 4.0提供`#pragma omp cancel for`這個可以跳脫迴圈的功能。
* 使用`tbb:: concurrent_unordered_set`，即可不用設ciritcal section就可以完成concurrent的set結構。但還是需要設一個cirtical section完成BFS的queue，因為`tbb:: concurrent_queue`並沒有提供front()和pop()的功能。
* 其他也有將能平行化的部分盡量平行化：程式初始化互相獨立的宣告式。

## Optimization:
* 提前偵測Dead state，即可避免不必要的搜尋。
* 在`std::map/set`找element是否在容器中，基本上有兩種方法，可以使用`set.find() == set.end()`或是`set.count()`，而前者是比較有效率的方法。

## Discussion:
* 即使程式做了平行化或是coding優化，還是沒有演算法層面來的強大。
* 有的程式做了平行化效能反而下降。

## Homework review:
* 學習到了openMp的用法，以前沒有嘗試寫過平行程式，openMp是個非常好上手的工具。
* 了解到平行程式需要良好的程式結構，在撰寫sequencial程式時就需要考量到這點。
* 學到了Boost Library有許多好用的函式。
* 學到了concurrent container的用法。

## References:
[1] https://rosettacode.org/wiki/Sokoban#
[2] https://baldur.iti.kit.edu/groupeffort/files/sokobanPortfolio.pdf
[3] https://stackoverflow.com/questions/25490357/checking-for-existence-in-stdmap-count-vs-find
[4] https://stackoverflow.com/questions/935467/parallelization-pthreads-or-openmp


