import torch
from torch import nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力模块"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.softmax=nn.Softmax(-1) # 被softmax的张量维度：(batch_size,n_heads,seq_len_Q,seq_len_K)
        self.dropout=nn.Dropout(dropout) # 随机失活部分参数，避免过拟合

    def forward(self, Q, K, V, mask=None):
        '''
        输入张量维度:(batch_size,n_heads,seq_len,head_dim)
        必须保证Q和K的特征维度相等:head_dim_Q=head_dim_K
        必须保证K和V的序列长度相等:seq_len_K=seq_len_V
        '''
        head_dim_K=Q.size(-1)
        # Q:(batch_size,n_heads,seq_len_Q,head_dim_Q)
        # K.transpose(-2,-1):(batch_size,n_heads,head_dim_K,seq_len_K)
        dot_product=torch.matmul(Q, K.transpose(-2, -1)) # (batch_size,n_heads,seq_len_Q,seq_len_K)
        dot_product=dot_product/math.sqrt(head_dim_K)
        if mask is not None:
            dot_product=dot_product.masked_fill(mask==0, float('-inf'))
        att_score=self.softmax(dot_product) # 在seq_len_K维度上归一化的注意力分数
        att_score=self.dropout(att_score)
        # att_score:(batch_size,n_heads,seq_len_Q,seq_len_K)
        # V:(batch_size,n_heads,seq_len_V,head_dim_V)
        result=torch.matmul(att_score, V) # 输出为att_score@Value的加权求和
        # result:(batch_size,n_heads,seq_len_Q,head_dim_V)
        return result, att_score # 输出att_score用于可视化

class MultiHeadAttention(nn.Module):
    """
    多头注意力模块
    融合Add&Norm层
    """
    def __init__(self, embed_dim=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim%n_heads==0 # 必须保证特征维度可被注意力头数整除
        # 如果Q/K和V的维度不等，需传入分别能被n_heads整除的embed_dim_Q/K和embed_dim_V，这里默认相等
        self.n_heads=n_heads
        self.head_dim=embed_dim//n_heads
        self.W_q=nn.Linear(embed_dim, embed_dim)
        self.W_k=nn.Linear(embed_dim, embed_dim)
        self.W_v=nn.Linear(embed_dim, embed_dim)
        self.att_block=ScaledDotProductAttention(dropout)
        self.W_o=nn.Linear(embed_dim, embed_dim) # 多头输出拼接后的全连接层，用于信息融合
        self.dropout=nn.Dropout(dropout) # 随机失活部分参数，避免过拟合
        self.norm=nn.LayerNorm(embed_dim)

    def forward(self, Q, K, V, mask=None):
        '''
        输入张量维度:(batch_size,seq_len,embed_dim)
        '''
        batch_size=Q.size(0)
        # 线性变换并拆分，再转置为点积注意力模块的输入格式(batch_size,n_heads,seq_len,head_dim)
        q=self.W_q(Q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k=self.W_k(K).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v=self.W_v(V).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # 传入点积注意力模块，得到维度为(batch_size,n_heads,seq_len_Q,head_dim_V)的输出
        result, _=self.att_block(q, k, v, mask)
        # 拼接多头的输出，contiguous()使tensor在内存中连续存储，避免view()报错
        # result:(batch_size,seq_len_Q,embed_dim_V)
        result=result.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads*self.head_dim)
        result=self.W_o(result)
        result=self.dropout(result)
        # 残差连接:Add&Norm
        output=self.norm(Q+result) # 无论self_att或cross_att，与输出建立残差连接的输入均为未经W_q变换的Q
        return output

class FeedForwardNetwork(nn.Module):
    """FFN前馈全连接层"""
    def __init__(self, embed_dim=512, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.ffn_block=nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout), # 中间激活层后随机失活
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm=nn.LayerNorm(embed_dim) # 归一化

    def forward(self, input):
        output=self.ffn_block(input)
        output=self.norm(input+output)
        return output
    
class PositionalEncoder(nn.Module):
    """位置编码"""
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe=torch.zeros(max_len, embed_dim)
        position=torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 1/10000.0**(2*i/embed_dim)=exp((2*i/embed_dim)*-ln(10000.0))
        div_term=torch.exp(torch.arange(0, embed_dim, 2).float()/embed_dim*(-math.log(10000.0)))
        pe[:, 0::2]=torch.sin(position*div_term)
        pe[:, 1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len=x.size(1)
        return x+self.pe[:, :seq_len, :]

class EncoderLayer(nn.Module):
    """编码器的一层"""
    def __init__(self, embed_dim=512, n_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.self_att=MultiHeadAttention(embed_dim, n_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)

    def forward(self, src, src_mask=None):
        self_att_output=self.self_att(src, src, src, src_mask)
        encoder_layer_output=self.ffn(self_att_output)
        return encoder_layer_output
    
class Encoder(nn.Module):
    """完整的编码器"""
    def __init__(self, vocab_size, n_encoder_layers=6, embed_dim=512, n_heads=8, ffn_dim=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size, embed_dim)
        self.positional_encoder=PositionalEncoder(embed_dim, max_len)
        self.encoder_layers=nn.ModuleList([EncoderLayer(embed_dim, n_heads, ffn_dim, dropout) for _ in range(n_encoder_layers)])

    def forward(self, src, src_mask=None):
        x=self.embedding(src)*math.sqrt(self.embedding.embedding_dim) # 缩放*sqrt(embed_dim)，方便后续注意力计算
        x=self.positional_encoder(x)
        for layer in self.encoder_layers:
            x=layer(x, src_mask)
        return x
    
class DecoderLayer(nn.Module):
    """解码器的一层"""
    def __init__(self, embed_dim=512, n_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.self_att=MultiHeadAttention(embed_dim, n_heads, dropout)
        self.cross_att=MultiHeadAttention(embed_dim, n_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        self_att_output=self.self_att(tgt, tgt, tgt, tgt_mask)
        cross_att_output=self.cross_att(self_att_output, memory, memory, memory_mask)
        decoder_layer_output=self.ffn(cross_att_output)
        return decoder_layer_output
    
class Decoder(nn.Module):
    """完整的解码器"""
    def __init__(self, vocab_size, n_decoder_layers=6, embed_dim=512, n_heads=8, ffn_dim=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size, embed_dim)
        self.positional_encoder=PositionalEncoder(embed_dim, max_len)
        self.decoder_layers=nn.ModuleList([DecoderLayer(embed_dim, n_heads, ffn_dim, dropout) for _ in range(n_decoder_layers)])
        # 将embed_dim映射到vocab_size，得到每个token的预测分布logits
        self.fc_output=nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x=self.embedding(tgt)*math.sqrt(self.embedding.embedding_dim) # 缩放*sqrt(embed_dim)，方便后续注意力计算
        x=self.positional_encoder(x)
        for layer in self.decoder_layers:
            x=layer(x, memory, tgt_mask, memory_mask)
        pred_logits=self.fc_output(x)
        return pred_logits
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, n_encoder_layers=6, n_decoder_layers=6, embed_dim=512, n_heads=8, ffn_dim=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.encoder=Encoder(src_vocab_size, n_encoder_layers, embed_dim, n_heads, ffn_dim, dropout, max_len)
        self.decoder=Decoder(tgt_vocab_size, n_decoder_layers, embed_dim, n_heads, ffn_dim, dropout, max_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory=self.encoder(src, src_mask)
        pred_proba=self.decoder(tgt, memory, tgt_mask, memory_mask)
        return pred_proba