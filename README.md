# Development Environment
Linux  
Python 3.8.17  
Pytorch

# Download Data
Download two datasets:  
1. MSMT17  
2. Market-1501  

Download TCMM pretrained:  
https://drive.google.com/file/d/1CcLWxjS4HcDgDL0bJ1n0B3CuB9bN_p5Z/view?usp=sharing  

# Path Setting 
### Modify the following code to match your data path.
TCMM pretrained path:  
`checkpoint = load_checkpoint(osp.join(args.logs_dir, '512_K4_r0.075_outlers.pth.tar'))`

GPU setting:  
`parser.add_argument('--gpu', type=str, default='0,1,2,3')`

batch size setting:  
`parser.add_argument('-b', '--batch-size', type=int, default=2048)`

data folder path:  
`arser.add_argument('--data-dir', type=str, metavar='PATH', default='/home/andy/ICASSP_data/data/')` 

# Evaluate
python ./train/evaluate.py

# Visualization
### T-SNE visualization
python ./train/vis_t-sne.py  

Reference: https://github.com/pavlin-policar/openTSNE/tree/master  

### Attention map visualization
python ./train/evaluate_heatmap.py  

Reference: https://github.com/facebookresearch/dino/blob/main/visualize_attention.py  
./train/TCMM/dino-main/visualize_attention.py  

Related:  
./train/evaluate_heatmap.py  

./train/TCMM/evaluators_heatmap.py:  
input batch data, file name list and model to vis_attention  
https://github.com/andy412510/TCMM/blob/b206973dc8e8511ebde93323e188dd59fcd94176/train/TCMM/evaluators_heatmap.py#L63  
follow reference work to obtain attention map, set path and patch here:  
https://github.com/andy412510/TCMM/blob/b206973dc8e8511ebde93323e188dd59fcd94176/train/TCMM/evaluators_heatmap.py#L21  

./train/TCMM/models/vision_transformer_heatmap.py  
### Ranking list visualization
This method is already integrated into the `Evaluate` program.

Reference: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/reidtools.py
./train/TCMM/evaluation_metrics/ranking_list_vis.py

Related:
./TCMM/evaluation_metrics/ranking.py

./TCMM/evaluation_metrics/ranking.py:
input batch id, top5 indices/matches list, query/gallery image path to top5_plot
https://github.com/andy412510/TCMM/blob/b206973dc8e8511ebde93323e188dd59fcd94176/train/TCMM/evaluation_metrics/ranking.py#L116

./train/TCMM/evaluation_metrics/ranking_list_vis.py:
Obtain ranking list visualization from reference work, set the visualization details and format here:
https://github.com/andy412510/TCMM/blob/c7c617e1224fac14418a2130594eb6318a12fa33/train/TCMM/evaluation_metrics/ranking_list_vis.py#L23
