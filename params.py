import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default="NELL-One", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-path", "--data_path", default="./NELL_1", type=str)  # ["./NELL", "./Wiki"]
    args.add_argument("-form", "--data_form", default="Pre-Train", type=str)  # ["Pre-Train", "In-Train", "Discard"]
    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument("-few", "--few", default=1, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=1024, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)	# 0.001
    args.add_argument("-es_p", "--early_stopping_patience", default=10, type=int)

    args.add_argument("-epo", "--epoch", default=100000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)

    args.add_argument("-b", "--beta", default=5, type=float)	# 5
    args.add_argument("-gm", "--gama", default=0.5, type=float)
    args.add_argument("-m", "--margin", default=1.0, type=float)	# default: 1
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-abla", "--ablation", default=False, type=bool)

    args.add_argument("-gpu", "--device", default=0, type=int)

    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)
    args.add_argument("-embed_model", "--embed_model", default="TransE", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-max_neighbor", "--max_neighbor", default=50, type=int)

    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.dataset == 'NELL-One':
        params['embed_dim'] = 100
    elif args.dataset == 'Wiki-One':
        params['embed_dim'] = 50

    params['device'] = torch.device('cuda:'+str(args.device))
#     params['device'] = 'cpu'

    return params


data_dir = {
    'train_tasks_in_train': '/train_tasks_in_train.json',
    'train_tasks': '/train_tasks.json',
    'test_tasks': "/test_tasks.json",
    'dev_tasks': "/dev_tasks.json",
    'train_tasks_woboost': '/train.json',
    'test_tasks_woboost': "/test.json",
    'dev_tasks_woboost': "/dev.json",
    'train_tasks_pull': '/train_tasks_pull.json',
    'dev_tasks_pull': '/dev_tasks_pull.json',
    'hand_sim':'/hand_sim.json',

    'rel2candidates_in_train': '/rel2candidates_in_train.json',
    'rel2candidates': '/rel2candidates.json',

    'e1rel_e2_in_train': '/e1rel_e2_in_train.json',
    'e1rel_e2': '/e1rel_e2.json',

    'ent2ids': '/ent2ids',
    'ent2vec': '/ent2vec.npy',
    'rel2ids': '/rel2ids',
    'train_id2rel':'/train_id2rel.json',
    'dev_id2rel': '/dev_id2rel.json',
    'train_key':'/train_key.json',
    'dev_key':'/dev_key.json'
}
