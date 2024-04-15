# -*- coding: utf-8 -*-
# @Time   : 2020/9/14
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
CFKG
##################################################
Reference:
    Qingyao Ai et al. "Learning heterogeneous knowledge base embeddings for explainable recommendation." in MDPI 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class CFKG(KnowledgeRecommender):
    r"""CFKG is a knowledge-based recommendation model, it combines knowledge graph and the user-item interaction
    graph to a new graph. In this graph, user, item and related attribute are viewed as entities, and the interaction
    between user and item and the link between item and attribute are viewed as relations. It define a new score
    function as follows:

    .. math::
        d (u_i + r_{buy}, v_j)

    Note:
        In the original paper, CFKG puts recommender data (u-i interaction) and knowledge data (h-r-t) together
        for sampling and mix them for training. In this version, we sample recommender data
        and knowledge data separately, and put them together for training.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset,dataset_):
        super(CFKG, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.loss_function = config['loss_function']
        self.margin = config['margin']
        assert self.loss_function in ['inner_product', 'transe']

        # csrec
        self.dataset_ = dataset_
        self.dataset = dataset
        self.temp = config['temperature']
        self.n_entities_ = dataset_.num(self.ENTITY_ID)
        self.n_relations_ = dataset_.num(self.RELATION_ID)
        self.n_items_ = dataset_.num(self.ITEM_ID)
        self.n_users_ = dataset_.num(self.USER_ID)
        self.n_share_entity = config["n_share_entity"]
        self.align_index_ = list(range(self.n_entities_-self.n_share_entity,self.n_entities_))
        self.share_entity = self.dataset_.field2id_token["entity_id"][self.align_index_]
        self.align_index = []
        for entity in self.share_entity:
            self.align_index.append(self.dataset.field2token_id["entity_id"][entity])
        assert len(self.align_index) == len(self.align_index_)
        assert (self.share_entity == self.dataset.field2id_token["entity_id"][self.align_index]).all()
        self.align_index = torch.tensor(self.align_index).to(self.device)
        self.align_index_ = torch.tensor(self.align_index_).to(self.device)
        self.align_label = torch.arange(0, len(self.align_index)).to(self.device)

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_embedding_ = nn.Embedding(self.n_users_, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_embedding_ = nn.Embedding(self.n_entities_, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations + 1, self.embedding_size)
        self.relation_embedding_ = nn.Embedding(self.n_relations_ + 1, self.embedding_size)
        if self.loss_function == 'transe':
            self.rec_loss = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')
        else:
            self.rec_loss = InnerProductLoss()
        self.align_loss = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.entity_embedding(item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)
        score = self._get_score(user_e, item_e, rec_r_e)
        return score

    def _get_rec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return user_e, pos_item_e, neg_item_e, rec_r_e
    
    def _get_rec_embedding_(self, user, pos_item, neg_item):
        user_e = self.user_embedding_(user)
        pos_item_e = self.entity_embedding_(pos_item)
        neg_item_e = self.entity_embedding_(neg_item)
        rec_r_e = self.relation_embedding_.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return user_e, pos_item_e, neg_item_e, rec_r_e

    def _get_kg_embedding(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_embedding(head)
        pos_tail_e = self.entity_embedding(pos_tail)
        neg_tail_e = self.entity_embedding(neg_tail)
        relation_e = self.relation_embedding(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e
    
    def _get_kg_embedding_(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_embedding_(head)
        pos_tail_e = self.entity_embedding_(pos_tail)
        neg_tail_e = self.entity_embedding_(neg_tail)
        relation_e = self.relation_embedding_(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e

    def _get_score(self, h_e, t_e, r_e):
        if self.loss_function == 'transe':
            return -torch.norm(h_e + r_e - t_e, p=2, dim=1)
        else:
            return torch.mul(h_e + r_e, t_e).sum(dim=1)
    
    def _get_align_loss(self,align_emb,align_emb_):
        # align_emb = F.normalize(align_emb, p=2, dim=1)
        # align_emb_ = F.normalize(align_emb_, p=2, dim=1)
        sim_score = torch.matmul(align_emb, align_emb_.transpose(0,1))
        sim_score = sim_score / self.temp
        loss = self.align_loss(sim_score,self.align_label)
        return loss
    
    def calculate_loss(self, interaction,interaction_):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        user_ = interaction_[self.USER_ID]
        pos_item_ = interaction_[self.ITEM_ID]
        neg_item_ = interaction_[self.NEG_ITEM_ID]
        
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]
        head_ = interaction_[self.HEAD_ENTITY_ID]
        relation_ = interaction_[self.RELATION_ID]
        pos_tail_ = interaction_[self.TAIL_ENTITY_ID]
        neg_tail_ = interaction_[self.NEG_TAIL_ENTITY_ID]

        user_e, pos_item_e, neg_item_e, rec_r_e = self._get_rec_embedding(user, pos_item, neg_item)
        user_e_, pos_item_e_, neg_item_e_, rec_r_e_ = self._get_rec_embedding_(user_, pos_item_, neg_item_)
        head_e, pos_tail_e, neg_tail_e, relation_e = self._get_kg_embedding(head, pos_tail, neg_tail, relation)
        head_e_, pos_tail_e_, neg_tail_e_, relation_e_ = self._get_kg_embedding_(head_, pos_tail_, neg_tail_, relation_)

        h_e = torch.cat([user_e, user_e_, head_e,head_e_])
        r_e = torch.cat([rec_r_e, rec_r_e_,relation_e,relation_e_])
        pos_t_e = torch.cat([pos_item_e, pos_item_e_, pos_tail_e,pos_tail_e_])
        neg_t_e = torch.cat([neg_item_e, neg_item_e_, neg_tail_e,neg_tail_e_])

        loss = self.rec_loss(h_e + r_e, pos_t_e, neg_t_e)

        align_emb = self.entity_embedding(self.align_index) # n_cate * emb_size
        align_emb_ = self.entity_embedding_(self.align_index_) # n_cate * emb_size
        align_loss = self._get_align_loss(align_emb,align_emb_)

        return loss, align_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)


class InnerProductLoss(nn.Module):
    r"""This is the inner-product loss used in CFKG for optimization.
    """

    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        pos_score = torch.mul(anchor, positive).sum(dim=1)
        neg_score = torch.mul(anchor, negative).sum(dim=1)
        return (F.softplus(-pos_score) + F.softplus(neg_score)).mean()
