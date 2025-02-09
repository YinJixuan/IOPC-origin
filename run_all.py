import os
import sys

datasets = [
'agnewsdataraw-8000_trans_subst_10',
'searchsnippets_trans_subst_10',
'stackoverflow_trans_subst_10',
'biomedical_trans_subst_10',
'TS_trans_subst_10',
'T_trans_subst_10',
'S_trans_subst_10',
'tweet-original-order_trans_subst_10'
]

for data in datasets:
    if data == 'agnewsdataraw-8000_trans_subst_10':
        num_classes = 4
        classes = 4
        pre_step = 600
        lambda2 = 1000
    elif data == 'searchsnippets_trans_subst_10':
        num_classes = 8
        classes = 8
        pre_step = 600
        lambda2 = 0.7
    elif data == 'stackoverflow_trans_subst_10':
        num_classes = 20
        classes = 20
        pre_step = -1
        lambda2 = 1000
    elif data == 'biomedical_trans_subst_10':
        num_classes = 20
        classes = 20
        pre_step = -1
        lambda2 = 1000
    elif data == 'TS_trans_subst_10':
        num_classes = 152
        classes = 152
        pre_step = 600
        lambda2 = 1.1
    elif data == 'T_trans_subst_10':
        num_classes = 152
        classes = 152
        pre_step = 600
        lambda2 = 1.2
    elif data == 'S_trans_subst_10':
        num_classes = 152
        classes = 152
        pre_step = 600
        lambda2 = 1.1
    elif data == 'tweet-original-order_trans_subst_10':
        num_classes = 89
        classes = 89
        pre_step = 600
        lambda2 = 1.2
        max_iter = 1000
        second_stage = 700
        os.system(
            'CUDA_VISIBLE_DEVICES=0  python main.py --dataname {} --num_classes {} --classes {} --pre_step {} --lambda2 {} --second_stage {} --max_iter {}'.format(
                data, num_classes, classes, pre_step, lambda2, second_stage, max_iter))
        break
    else:
        sys.stderr.write("wrong dataset name")
        sys.exit(1)

    sys.stderr.write("\n**********************************************\n")
    sys.stderr.write("dataset: {}".format(data))
    sys.stderr.write("\n**********************************************\n")
    sys.stderr.flush()

    os.system(
        'CUDA_VISIBLE_DEVICES=0  python main.py --dataname {} --num_classes {} --classes {} --pre_step {} --lambda2 {}'.format(
            data, num_classes, classes, pre_step, lambda2))