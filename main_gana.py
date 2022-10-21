from trainer_gana import *
from params import *
from data_loader import *
import json

if __name__ == '__main__':
    params = get_params()

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path']+v

    tail = '_in_train'
    tail_ = '_pull'
    #if params['data_form'] == 'In-Train':
    #    tail = '_in_train'

    dataset = dict()
    print("loading train_tasks{} ... ...".format(tail_))
    dataset['train_tasks'] = json.load(open(data_dir['train_tasks'+tail_]))
    print("loading test_tasks ... ...")
    dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks'+tail_]))
    print("loading rel2candidates{} ... ...".format(tail))
    dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates'+tail]))
    print("loading e1rel_e2{} ... ...".format(tail))
    dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2'+tail]))
    print("loading ent2id ... ...")
    dataset['ent2id'] = json.load(open(data_dir['ent2ids']))
    dataset['rel2id'] = json.load(open(data_dir['rel2ids']))
    dataset['hand_sim'] = json.load(open(data_dir['hand_sim']))
    dataset['train_id2rel'] = json.load(open(data_dir['train_id2rel']))
    dataset['dev_id2rel'] = json.load(open(data_dir['dev_id2rel']))
    dataset['train_key'] = json.load(open(data_dir['train_key']))
    dataset['dev_key'] = json.load(open(data_dir['dev_key']))

    if params['data_form'] == 'Pre-Train':
        print('loading embedding ... ...')
        dataset['ent2emb'] = np.loadtxt(params['data_path']+'/entity2vec.TransE')
        dataset['rel2emb'] = np.loadtxt(params['data_path']+'/relation2vec.TransE')

    print("----------------------------")

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]

    # trainer
    trainer = Trainer(data_loaders, dataset, params)

    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True)
    elif params['step'] == 'test':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=True)
        else:
            trainer.eval(istest=True)
    elif params['step'] == 'dev':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=False)
        else:
            trainer.eval(istest=False)

