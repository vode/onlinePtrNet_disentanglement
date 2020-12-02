import torch
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
class DataHelper():
    def __init__(self,embedding_path):
        self.id_to_token = []
        self.token_to_id = {}
        # Create word embeddings and initialise
        self.id_to_token = []
        self.token_to_id = {}
        self.pretrained = []
        for line in open(embedding_path):
            parts = line.strip().split()
            word = parts[0].lower()
            vector = [float(v) for v in parts[1:]]
            self.token_to_id[word] = len(self.id_to_token)
            self.id_to_token.append(word)
            self.pretrained.append(vector)
        self.NWORDS = len(self.id_to_token)
        self.DIM_WORDS = len(self.pretrained[0])
    def sent2index(self,sent,max_seq = 32):
        index = [0 for i in range(max_seq)]
        unk_id = self.token_to_id.get('<unka>',0)
        for i,word in enumerate(sent):
            if i >= max_seq:
                break
            id = self.token_to_id.get(word,unk_id)
            index[i] = id
        return index
class EsimEmbedder(nn.Module):
    def __init__(self, embeds_dim,hidden_size,num_word,weight_matrix):
        super(EsimEmbedder, self).__init__()
        self.dropout = 0.2
        linear_size = 128
        self.hidden_size = hidden_size
        self.embeds_dim = embeds_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weight = torch.FloatTensor(weight_matrix)
        weight.to(self.device)
        self.embeds = nn.Embedding.from_pretrained(weight)
        self.embeds.to(self.device)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.GRU(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=False)
        self.lstm2 = nn.GRU(self.hidden_size*4, self.hidden_size, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
                nn.BatchNorm1d(self.hidden_size * 4),
                nn.Linear(self.hidden_size * 4, linear_size),
                nn.ELU(inplace=True),
                nn.BatchNorm1d(linear_size),
                nn.Dropout(self.dropout),
                nn.Linear(linear_size, linear_size),
                nn.ELU(inplace=True),
                nn.BatchNorm1d(linear_size),
                nn.Dropout(self.dropout),
            )
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))
        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention , dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) , dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size
        return x1_align, x2_align
    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)
    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)
    def forward(self, input):
        # print(input)
        # batch_size * seq_len
        sent1, sent2 = torch.tensor(input[0]).to(self.device), torch.tensor(input[1]).to(self.device)
        mask1, mask2 = sent1.eq(0), sent2.eq(0)
        mask1.to(self.device)
        mask2.to(self.device)
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)
        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)
        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)
        sen1 = torch.mean(q1_align,1)
        sen2 = torch.mean(q2_align,1)
        # return self.fc(torch.cat([sen1,sen2,self.submul(sen1,sen2)],-1))
        # print(q1_align.shape)
        # return torch.cat([sen1,sen2],-1)
        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)
        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)
        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)
        # feature_vectors
        x = torch.cat([q1_rep, q2_rep], -1)
        x = self.fc(x)
        return x
# Joint Model is to model both reply-to relationship and same-conversation relationship
class JointModel(torch.nn.Module):
    def __init__(self,FEATURES,HIDDEN,replyModel,cluster_weight = 0.1):
        super().__init__()
        #
        self.cluster_weight = cluster_weight
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        feature_size = FEATURES+128
        # feature_size = 128
        self.hidden1 = torch.nn.Linear(feature_size, HIDDEN)
        self.nonlin1 = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.nonlin2 = torch.nn.ReLU()
        self.hidden3 = torch.nn.Linear(HIDDEN,1)
        self.nonlin3 = torch.nn.Sigmoid()
        self.softmax_loss = torch.nn.BCELoss(reduction='sum')
        self.replyModel = replyModel
    def forward(self, query, options, gold, lengths, query_no, gold_cluster):
        final_features,loss, predicted_link, max_score,score_list = self.replyModel(query, options, gold, lengths, query_no,True)
        h1 = self.nonlin1(self.hidden1(final_features))
        h2 = self.nonlin2(self.hidden2(h1))
        scores = self.nonlin3(self.hidden3(h2))
        output_scores = torch.unsqueeze(torch.unsqueeze(scores, 0), 2)
        true_out = torch.tensor(gold_cluster).float().to(self.device)
        true_out = torch.unsqueeze(true_out,-1)
        cluster_loss = self.softmax_loss(input=scores, target=true_out)
        cluster_list = np.array(torch.squeeze(output_scores).data.tolist())
        return loss + self.cluster_weight*cluster_loss, predicted_link,max_score, score_list,cluster_list
    def get_ids(self, words):
        return self.replyModel.get_ids(words)
# This is to model reply-to relationship
class PyTorchModel(torch.nn.Module):
    def __init__(self,FEATURES,HIDDEN):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_helper = DataHelper("./data/glove-ubuntu.txt")
        self.esim_embedder = EsimEmbedder(hidden_size=128,embeds_dim=self.data_helper.DIM_WORDS,num_word = self.data_helper.NWORDS,weight_matrix=self.data_helper.pretrained)
        feature_size = FEATURES+128
        self.hidden1 = torch.nn.Linear(feature_size, HIDDEN)
        self.nonlin1 = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.nonlin2 = torch.nn.ReLU()
        self.hidden3 = torch.nn.Linear(HIDDEN,1)
        self.nonlin3 = torch.nn.Sigmoid()
        self.softmax_loss = torch.nn.BCELoss(reduction='sum')
        self.margin_loss = torch.nn.MultiLabelMarginLoss(reduction='sum')
        self.l1_loss = torch.nn.SmoothL1Loss(reduction='sum')
        self.esim_embedder.to(self.device) 
        # self.esim_embedder.cuda()
    def get_ids(self, words):
        return self.data_helper.sent2index(words,max_seq=64)
    def forward(self, query, options, gold, lengths, query_no,return_feature = False):
        answer = max(gold)
        gold_label = [0.0] * len(options)
        weights = []
        relative_scores = []
        for i in gold:
            gold_label[i] = 1.0
        # Concatenate the other features
        features = torch.tensor([v[1] for v in options]).to(self.device)
        weights = torch.tensor(weights).to(self.device)
        relative_scores = torch.tensor(relative_scores).to(self.device)
        opt_tok = [v[0] for v in options]
        query_tok =  [query for v in range(len(options))]
        sent_feature = self.esim_embedder([query_tok,opt_tok])
        final_features = torch.cat([features,sent_feature],-1)
        # final_features = sent_feature
        # print(final_features.shape)
        h1 = self.nonlin1(self.hidden1(final_features))
        h2 = self.nonlin2(self.hidden2(h1))
        h3 = self.nonlin3(self.hidden3(h2))
        #print(h2.shape)
        scores = torch.sum(h2, 1)
        scores = torch.softmax(scores,-1)
        scores = torch.unsqueeze(scores,-1)
        #print(scores.shape)
        output_scores = torch.unsqueeze(torch.unsqueeze(scores, 0), 2)
        #print(output_scores.shape)
        # Get loss and prediction
        true_out = torch.tensor(gold_label).float().to(self.device)
        true_out = torch.unsqueeze(true_out,-1)
        #print(output_scores.shape)
        # print(scores.shape,true_out.shape)
        softmax_loss = self.softmax_loss(input=scores, target=true_out)
        #margin_loss = self.margin_loss(input = scores.transpose(-2,-1),target=true_out.transpose(-2,-1).long())
        #regression_loss = self.l1_loss(input = scores,target = relative_scores)
        loss = softmax_loss
        predicted_link = torch.argmax(output_scores, 1)
        score = torch.max(output_scores,1)[0].data.tolist()[0][0][0]
        if return_feature:
            return final_features,loss, predicted_link, score, np.array(torch.squeeze(output_scores).data.tolist())
        else:
            return loss, predicted_link, score, np.array(torch.squeeze(output_scores).data.tolist())
    # triplet loss
    def calc_loss(self,out_scores,gold):
        i = gold
        neg_scores = torch.cat([out_scores[0:i], out_scores[i+1:]])
        pos_score = out_scores[i]
        max_sim_neg = torch.max(neg_scores)
        #print(type(max_sim_neg))
        loss = torch.mean(torch.clamp(1 - pos_score + max_sim_neg, min=0))
        return loss