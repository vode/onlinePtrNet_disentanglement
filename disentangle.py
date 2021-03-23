#!/usr/bin/env python3
import argparse
import random
import sys
import string
import time
import numpy as np
import pickle
from util import *
def header(args, out=sys.stdout):
    head_text = "# "+ time.ctime(time.time())
    head_text += "\n# "+ ' '.join(args)
    for outfile in out:
        print(head_text, file=outfile)
parser = argparse.ArgumentParser(description='Disentangler.')
parser.add_argument('prefix')
parser.add_argument('--train', nargs="+")
parser.add_argument('--dev', nargs="+")
parser.add_argument('--test', nargs="+")
parser.add_argument('--model')
parser.add_argument('--cluster_model',default="cluster_match")
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--word-vectors')
parser.add_argument('--max-dist', default=101, type=int)
parser.add_argument('--random-sample')
parser.add_argument('--test-start', type=int)
parser.add_argument('--test-end', type=int)
parser.add_argument('--report-freq', default=5000, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--weight-decay', default=1e-8, type=float)
parser.add_argument('--hidden', default=1024, type=int)#64)
parser.add_argument('--learning-rate', default=0.1, type=float)
parser.add_argument('--learning-decay-rate', default=0.0, type=float)
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--drop', default=0.0, type=float)
parser.add_argument('--layers', default=2, type=int)
parser.add_argument('--clip', default=1.0, type=float)
args = parser.parse_args()
WEIGHT_DECAY = args.weight_decay
HIDDEN = args.hidden
LEARNING_RATE = args.learning_rate
LEARNING_DECAY_RATE = args.learning_decay_rate
MOMENTUM = args.momentum
EPOCHS = args.epochs
DROP = args.drop
MAX_DIST = args.max_dist
log_file = open(args.prefix +".log", 'w')
header(sys.argv, [log_file, sys.stdout])
import torch
from model import PyTorchModel,JointModel
torch.manual_seed(args.seed)
test_cluster_predict = {
    "tp":0,
    "fp":0,
    "tn":0,
    "fn":0
}

def do_instance(instance, train, model, optimizer, do_cache=True,shift = False,cluser_info =False):
    name, query, gold, text_ascii, text_tok, info, target_info,channel = instance
    # Skip cases if we can't represent them
    gold = [v for v in gold if v > query - MAX_DIST]
    # print(len(gold),train)
    if len(gold) == 0 :
        return 0, False, query,0.0,None,None
    if train:
        cluster = train_cluster
    else:
        cluster = test_cluster
    gold_cluster = []
    # Get features
    options = []
    query_ascii = text_ascii[query]
    # if gold[0] == query:
    #     return 0, False,0
    query_tok = model.get_ids(text_tok[query])
    distance = []
    # shift controls whether consider self-link in the prediction
    if shift:
        for i in range(query-1, max(-1, query - MAX_DIST -1 ), -1):
            #d = channel.distance(query,i)
            distance.append(1)
            option_ascii = text_ascii[i]
            option_tok = model.get_ids(text_tok[i])
            features = feature_map[(name,query,i)] 
            options.append((option_tok, features))
            gold_cluster.append(0.0)
    else:
        for i in range(query, max(-1, query - MAX_DIST ), -1):
            #d = channel.distance(query,i)
            distance.append(1)
            option_ascii = text_ascii[i]
            option_tok = model.get_ids(text_tok[i])
            features = feature_map[(name,query,i)]
            options.append((option_tok, features))
            if query in cluster[name] and i in cluster[name]:
                if cluster[name][query] == cluster[name][i]:
                    gold_cluster.append(1.0)
                else:
                    gold_cluster.append(0.0)
            else:
                gold_cluster.append(0)
    gold = [query - v for v in gold]
    # for item in gold:
    #     assert distance[item] == 1
    lengths = [len(sent) for sent in options]
    # Run computation
    example_loss, output, score, parent_score_list,cluster_score_list = model(query_tok, options, gold, lengths, query, gold_cluster)
    loss = 0.0
    if train and example_loss is not None:
        with torch.autograd.detect_anomaly():
            example_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()
            # optimizer.zero_grad()
            loss = example_loss.item()
    predicted = output
    # if predicted == 0:
    #     predicted +=1
    predicted = output.item()
    matched = (predicted in gold)
    if not train and not shift:
        for index, score in enumerate(cluster_score_list):
            if index != 0:
                if 1.0 == gold_cluster[index]:
                    if score > 0.5:
                        test_cluster_predict['tp'] +=1
                    else:
                        test_cluster_predict['fn'] +=1
                else:
                    if score > 0.5:
                        test_cluster_predict['fp'] +=1
                    else:
                        test_cluster_predict['tn'] +=1
    if shift:
        return loss, matched, predicted+1,score,parent_score_list,cluster_score_list
    else:
        return loss, matched, predicted, score,parent_score_list,cluster_score_list
###############################################################################

train = []
# train,dev,test is the labeled reply to data
# train_cluster, dev_cluster, test_cluster is the labeled same conversation data
if args.train:
    train,train_stat,train_cluster = read_data(args.train)
dev = []
if args.dev:
    dev,dev_stat,dev_cluster = read_data(args.dev)
    test = dev
    test_stat = dev_stat
    test_cluster = dev_cluster
if args.test:
    test,test_stat,test_cluster = read_data(args.test)
    
with open('data/processed_data.pkl','rb') as input:
    feature_map = pickle.load(input)              

# Model and optimizer creation
model = None
optimizer = None
scheduler = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
replyModel = PyTorchModel(FEATURES,HIDDEN)
# if not torch.cuda.is_available():
#     replyModel.load_state_dict(torch.load("esim.pt.model",map_location=torch.device('cpu')))
# else:
#     replyModel.load_state_dict(torch.load("esim.pt.model"))
model = JointModel(FEATURES,HIDDEN,replyModel)    
# model = PyTorchModel(FEATURES,HIDDEN)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY)
rescale_lr = lambda epoch: 1 / (1 + LEARNING_DECAY_RATE * epoch)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=rescale_lr)
prev_best = None
if args.train:
    step = 0
    for epoch in range(EPOCHS):
        random.shuffle(train)
        # Loop over batches
        loss = 0
        match = 0
        total = 0
        loss_steps = 0
        for instance in train:
            step += 1
            model.train() 
            optimizer.zero_grad()
            ex_loss, matched, _ ,score,_,_ = do_instance(instance, True, model, optimizer)
            loss += ex_loss
            loss_steps += 1
            if matched:
                match += 1
            total += len(instance[2])
            # Partial results
            if step % args.report_freq == 0:
                # Dev pass         
                model.eval() 
                dev_match = 0
                dev_total = 0
                for dinstance in dev:
                    _, matched, _,_,_,_ = do_instance(dinstance, False, model, optimizer)
                    if matched:
                        dev_match += 1
                    dev_total += len(dinstance[2])
                tacc = match / total
                dacc = dev_match / dev_total
                print("{} tl {:.3f} ta {:.3f} da {:.3f} from {} {}".format(epoch, loss / loss_steps, tacc, dacc, dev_match, dev_total), file=log_file)
                log_file.flush()
                if prev_best is None or prev_best[0] < dacc:
                    prev_best = (dacc, epoch)
                    torch.save(model.state_dict(), args.prefix +".pt.model")
                        # torch.save(model.esim_embedder.state_dict(), args.prefix +"esim.pt.model")
        scheduler.step()
        if prev_best is not None and epoch - prev_best[1] > 5:
            break
# Load model
if prev_best is not None or args.model:
    location = args.model
    if location is None:
        location = args.prefix +'.pt.model'
    #model.esim_embedder.load_state_dict(torch.load(args.prefix +'esim.pt.model'))
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(location,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(location))
def decoding(buffer,threshold = 0.90):
    confirm = []
    unconfirm = {}
    def get_score(u1,u2):
        _,score_list,_ = buffer[u1]
        try:
            if u1-u2 > len(score_list) -1:
                return -1
            else:
                return score_list[u1-u2]
        except Exception:
            return -1
    for u in buffer.keys():
        score_list, cluster_list,instance = buffer[u]
        if cluster_list is None:
            confirm.append((u,u))
            continue
        largest = u - np.argmax(score_list)
       # _,_,pred,_  = do_instance(instance, False, model, optimizer, False,True)
       # second_largest = u -pred
            #
        if largest == u:
            _,_,pred,_ , _,_ = do_instance(instance, False, model, optimizer, False,True)
            second_largest = u -pred
            if np.max(score_list) > threshold:
                confirm.append((u,largest))
            else:
                confirm.append((u,second_largest))
            #unconfirm[u]= (largest,second_largest,np.max(score_list))
        else:
            if largest in unconfirm:
                score_1 = get_score(u,unconfirm[largest][1])
                score_2 = get_score(u,unconfirm[largest][0])
                #print("scores")
                #print(score_1,score_2,unconfirm[largest][2])
                if score_1 > threshold or score_2 < threshold:
                    confirm.append((largest,unconfirm[largest][1]))
                    del unconfirm[largest]
            confirm.append((u,largest))
            #unconfirm[u] = largest,second_largest
    for k in unconfirm.keys():
        confirm.append((k,unconfirm[k][0]))
    s = sorted(confirm, key = lambda x: x[0])
    return s
predict_cluster = {}
current_buffer = {}
current_file = None
# Evaluation pass.
test_cluster_predict = {
    "tp":0,
    "fp":0,
    "tn":0,
    "fn":0
}
for instance in test:
    predict = 0.0
    file = instance[0]
    if file not in predict_cluster:
        predict_cluster[file]={}
    if file != current_file and current_file is not None:
        s = decoding(current_buffer)
        for item in s:
            print("{}:{} {} -".format(current_file, item[0], item[1]))
        # refresh buffer for new dialogue file
        current_buffer = {}
    current_file = file
    result = do_instance(instance, False, model, optimizer, False,False,True)
    name = instance[0]
    parent_score_array = result[-2]
    cluster_score_array = result[-1]
    current_buffer[instance[1]] = (parent_score_array,cluster_score_array,instance)
s = decoding(current_buffer)
for item in s:
    print("{}:{} {} -".format(current_file, item[0], item[1]))
#print(test_cluster_predict)
log_file.close()
# with open('processed_data','wb') as output:
#     pickle.dump(feature_map,output)
