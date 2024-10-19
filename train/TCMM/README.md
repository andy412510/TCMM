# Evaluators Process
### Input
1. test data loader
2. query data
3. gallery data

### 程式流程
https://github.com/andy412510/TCMM/blob/4c5c051a49836e3017d8bf72fd762943116a6a84/train/TCMM/evaluators.py#L118  
^^^^ 程式開始 evaluate 的第一行 ^^^^

https://github.com/andy412510/TCMM/blob/4c5c051a49836e3017d8bf72fd762943116a6a84/train/TCMM/evaluators.py#L119  
output: features = {Tensors:(93820,768)}  
代表所有 test data features，包含 11659 query and 82161 gallery 共 93820筆，每筆資料 768 維

https://github.com/andy412510/TCMM/blob/4c5c051a49836e3017d8bf72fd762943116a6a84/train/TCMM/evaluators.py#L120  
output: distmat = {Tensors:(11659,82161)}、query_features = {ndarray: (11659,768)}、gallery_features = {ndarray: (82161,768)}  
distmat: 代表每個 query，和 gallery 中所有資料的歐式距離  
query_features、gallery_features: query 和 gallery 資料的 features 轉成 numpy 格式

train/TCMM/evaluation_metrics/ranking.py  
https://github.com/andy412510/TCMM/blob/4c5c051a49836e3017d8bf72fd762943116a6a84/train/TCMM/evaluation_metrics/ranking.py#L101  
indices 保存的是每個 query 對應的 gallery 資料按距離（從小到大）排序後的 index

https://github.com/andy412510/TCMM/blob/4c5c051a49836e3017d8bf72fd762943116a6a84/train/TCMM/evaluation_metrics/ranking.py#L102  
matches 是一個布林矩陣，表示哪些 query 和 gallery 的 ID 是 match 的

這裡就是要麻煩你可視化 re-id ranking 的地方  
把 indices 裡面對應的 query 和距離最小的五個 gallery 畫出來  
並依照 matches 的資訊給 gallery 框框，綠框代表正確，紅框代表錯誤  
我目前 batch size 是 2048  
你可以設定成1說不定比較方便你實作  

  

