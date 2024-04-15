# -*- coding: utf-8 -*-
# @Time   : 2020/8/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
CKE
##################################################
Reference:
    Fuzheng Zhang et al. "Collaborative Knowledge Base Embedding for Recommender Systems." in SIGKDD 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class CKE(KnowledgeRecommender):
    r"""CKE is a knowledge-based recommendation model, it can incorporate KG and other information such as corresponding
    images to enrich the representation of items for item recommendations.

    Note:
        In the original paper, CKE used structural knowledge, textual knowledge and visual knowledge. In our
        implementation, we only used structural knowledge. Meanwhile, the version we implemented uses a simpler
        regular way which can get almost the same result (even better) as the original regular way.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset,dataset_):
        super(CKE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']
        self.reg_weights = config['reg_weights']
        
        # csrec
        self.dataset_ = dataset_
        self.dataset = dataset
        self.temp = config['temperature']
        self.n_entities_ = dataset_.num(self.ENTITY_ID)
        self.n_relations_ = dataset_.num(self.RELATION_ID)
        self.n_items_ = dataset_.num(self.ITEM_ID)
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
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_embedding_ = nn.Embedding(self.n_entities_, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.relation_embedding_ = nn.Embedding(self.n_relations_, self.kg_embedding_size)
        self.trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        # self.trans_w_ = nn.Embedding(self.n_relations_, self.embedding_size * self.kg_embedding_size)
        self.rec_loss = BPRLoss()
        self.kg_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.align_loss = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        r_e = F.normalize(r_e, p=2, dim=1)
        h_e = F.normalize(h_e, p=2, dim=1)
        pos_t_e = F.normalize(pos_t_e, p=2, dim=1)
        neg_t_e = F.normalize(neg_t_e, p=2, dim=1)

        return h_e, r_e, pos_t_e, neg_t_e, r_trans_w
    
    def _get_kg_embedding_(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding_(h).unsqueeze(1)
        pos_t_e = self.entity_embedding_(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding_(neg_t).unsqueeze(1)
        r_e = self.relation_embedding_(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)
        # r_trans_w = self.trans_w_(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        r_e = F.normalize(r_e, p=2, dim=1)
        h_e = F.normalize(h_e, p=2, dim=1)
        pos_t_e = F.normalize(pos_t_e, p=2, dim=1)
        neg_t_e = F.normalize(neg_t_e, p=2, dim=1)

        return h_e, r_e, pos_t_e, neg_t_e, r_trans_w

    def forward(self, user, item):
        u_e = self.user_embedding(user)
        i_e = self.item_embedding(item) + self.entity_embedding(item)
        score = torch.mul(u_e, i_e).sum(dim=1)
        return score

    def _get_rec_loss(self, user_e, pos_e, neg_e):
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        rec_loss = self.rec_loss(pos_score, neg_score)
        return rec_loss

    def _get_kg_loss(self, h_e, r_e, pos_e, neg_e):
        pos_tail_score = ((h_e + r_e - pos_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_e) ** 2).sum(dim=1)
        kg_loss = self.kg_loss(neg_tail_score, pos_tail_score)
        return kg_loss
    
    def _get_align_loss(self,align_emb,align_emb_):
        align_emb = F.normalize(align_emb, p=2, dim=1)
        align_emb_ = F.normalize(align_emb_, p=2, dim=1)
        sim_score = torch.matmul(align_emb, align_emb_.transpose(0,1))
        sim_score = sim_score / self.temp
        loss = self.align_loss(sim_score,self.align_label)
        return loss

    def calculate_loss(self, interaction,interaction_):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]
        h_ = interaction_[self.HEAD_ENTITY_ID]
        r_ = interaction_[self.RELATION_ID]
        pos_t_ = interaction_[self.TAIL_ENTITY_ID]
        neg_t_ = interaction_[self.NEG_TAIL_ENTITY_ID]

        user_e = self.user_embedding(user)
        pos_item_e = self.item_embedding(pos_item)
        neg_item_e = self.item_embedding(neg_item)
        pos_item_kg_e = self.entity_embedding(pos_item)
        neg_item_kg_e = self.entity_embedding(neg_item)
        pos_item_final_e = pos_item_e + pos_item_kg_e
        neg_item_final_e = neg_item_e + neg_item_kg_e

        rec_loss = self._get_rec_loss(user_e, pos_item_final_e, neg_item_final_e)

        h_e, r_e, pos_t_e, neg_t_e, r_trans_w = self._get_kg_embedding(h, r, pos_t, neg_t)
        h_e_, r_e_, pos_t_e_, neg_t_e_, r_trans_w_ = self._get_kg_embedding_(h_, r_, pos_t_, neg_t_)
        kg_loss = self._get_kg_loss(h_e, r_e, pos_t_e, neg_t_e)
        kg_loss_ = self._get_kg_loss(h_e_, r_e_, pos_t_e_, neg_t_e_)

        align_emb = self.entity_embedding(self.align_index) # n_cate * emb_size
        align_emb_ = self.entity_embedding_(self.align_index_) # n_cate * emb_size
        align_loss = self._get_align_loss(align_emb,align_emb_)

        reg_loss = self.reg_weights[0] * self.reg_loss(user_e, pos_item_final_e, neg_item_final_e) + \
                   self.reg_weights[1] * self.reg_loss(h_e, r_e, pos_t_e, neg_t_e) + \
                   self.reg_weights[1] * self.reg_loss(h_e_, r_e_, pos_t_e_, neg_t_e_) + \
                   self.reg_weights[1] * self.reg_loss(align_emb,align_emb_)

        return rec_loss, kg_loss, kg_loss_, align_loss, reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight + self.entity_embedding.weight[:self.n_items]
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
