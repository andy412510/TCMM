# Development Environment
Linux
Python 3.8.17

# Install
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  
# Donload Data
Choose one dataset source:
1. NAS: MVL_FTP/Dataset/Person re-id/MSMT17_V1.zip
2. 113: /home/andy/ICASSP_data/data/MSMT17

ViT pretrained:
1. NAS: Master Thesis/Thesis/105 朱政安/code/log/pass_vit_small_full.pth
2. 113: /home/andy/ICASSP_data/pretrain/PASS/pass_vit_small_full.pth

TCMM pretrained:
1. NAS: Master Thesis/Thesis/105 朱政安/code/log/msmt/512_K4_r0.075_outlers.pth.tar
2. /home/andy/main_code/train/log/cluster_contrast_reid/msmt17_v1/512_K4_r0.075_outlers.pth.tar

# Path Setting 
`checkpoint = load_checkpoint(osp.join(args.logs_dir, '512_K4_r0.075_outlers.pth.tar'))`  # TCMM pretrained path

`parser.add_argument('--gpu', type=str, default='0,1,2,3')`  # GPU setting

`parser.add_argument('-b', '--batch-size', type=int, default=2048)`  # batch size setting

`parser.add_argument('-pp', '--pretrained-path', type=str, default='/home/andy/ICASSP_data/pretrain/PASS/pass_vit_small_full.pth')`  # ViT pretrained path

`arser.add_argument('--data-dir', type=str, metavar='PATH', default='/home/andy/ICASSP_data/data/')`  # data folder path

	