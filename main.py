import torch
import torch.nn as nn
from encoders.clap_encoder import ClapEncoder
from layers.cross_attention import CrossAttention
from layers.self_attention import SelfAttention
from encoders.transformer_encoder import TransformerEncoderLayer
from layers.pooling_layer import PoolingLayer
from layers.dense_layer import DenseLayer
from layers.output_layer import OutputLayer

class CALM():
    def encoder_model(audio_path, text):
        model = ClapEncoder()
        audio_emb = model.get_text_embeddings(text)
        text_emb  = model.get_audio_embeddings(audio_path)
        return audio_emb, text_emb
    
    def CrossAtention(query,key,value): 
        cross_attn = CrossAttention() 
        output = cross_attn(query, key, value)
        return output
        
    def self_attention(query,key,value):
        self_attn = SelfAttention()
        output = self_attn(query, key, value)
        return output
    
    def transform_encoder(embeddings):
        encoder = TransformerEncoderLayer(embeddings)
        return encoder
    
    def Pooling_Layer(encode_emb):
        layer = PoolingLayer()
        output  = layer(encode_emb)
        return output
    
    def DenseLayer(pooled_emb):
        layer = DenseLayer()
        output = layer(pooled_emb)
    
    def Output(dense_emb):
        layer = OutputLayer()
        output = layer(dense_emb)
        return output

 