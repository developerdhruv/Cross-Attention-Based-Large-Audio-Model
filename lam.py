from encoders.clap_encoder import ClapEncoder
from encoders.transformer_encoder import TransformerEncoderLayer
from layers.dense_layer import DenseLayer
from layers.self_attention import SelfAttention
from layers.pooling_layer import PoolingLayer
from layers.output_layer import OutputLayer
from layers.cross_attention import CrossAttention 

import torch.nn as nn

class LAM(nn.Module):
    def __init__(self,):
        super(LAM, self).__init__()
        self.clap_encoder = ClapEncoder()  
        self.cross_attn = CrossAttention() 
        self.self_attn = SelfAttention()
        self.output_layer = OutputLayer()
        self.pooling_layer = PoolingLayer()
        self.dense_layer = DenseLayer()

    def forward(self, text,audio_path): 
        audio_emb = self.clap_encoder.get_audio_embeddings(audio_path) 
        text_emb = self.clap_encoder.get_text_embeddings(text)   
        cross_attn_output = self.cross_attn(text_emb,audio_emb,audio_emb ) 
        self_attn_output = self.self_attn(cross_attn_output,cross_attn_output,cross_attn_output) 
        dense_output = self.dense_layer(self_attn_output) 
        pooling_output = self.pooling_layer(dense_output) 
        output = self.output_layer(pooling_output)
        return output
         

        
    
        

