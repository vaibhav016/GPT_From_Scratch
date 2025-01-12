import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self, vocabulary_size=50000, 
                 embedding_size=768,  
                 sequence_len=512, 
                ):
        super(GPTEmbedding, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.sequence_len = sequence_len
        
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding = nn.Embedding(vocabulary_size, embedding_size) 
    
    def forward(self, tokens, positions):
        if torch.any(positions>=self.sequence_len):
            raise RuntimeError("Positions > Seq len")
        
        if torch.any(tokens>self.vocabulary_size):
            raise RuntimeError("Tokens not in Vocab")

        token_embedding = self.token_embedding(tokens.long())
        position_embedding = self.position_embedding(positions.long())

        return token_embedding + position_embedding


class MHA(nn.Module):
    def __init__(self, head_dim, num_heads, sequence_len):
        super(MHA, self).__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.sequence_len = sequence_len

        self.QW = nn.Linear(self.num_heads*self.head_dim, self.num_heads*self.head_dim) 
        self.KW = nn.Linear(self.num_heads*self.head_dim, self.num_heads*self.head_dim) 
        self.VW = nn.Linear(self.num_heads*self.head_dim, self.num_heads*self.head_dim) 

        self.WO = nn.Linear(self.num_heads*self.head_dim, self.num_heads*self.head_dim) 

    def split_heads(self, x):
        # [16 x 10 x 512] 
        # [16 x 10 x 4, 128] 
        batch_size, sequence_len, _ = x.size()
        x = x.view(batch_size, sequence_len, self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        # [16 x 4 x 10 x 128] 
        return x 

    def merge_heads(self, x):
        # [16 x 10 x 512] 
        # [16 x 4 x 10 x 128] 
        batch_size, nh, sequence_len, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, sequence_len, self.head_dim*self.num_heads)
        return x 
       
        # [16 x 10 x 512] 

    def apply_attention(self, queries, keys, values):
        # query -> B x S x E 
        #  Q [16 x 10 x 128] 
         # K [16 x 10 x 128] ->  [16 x 128 x 10].T 
        # Att_out => [16x 10 x 10] 
          
        attention = torch.matmul(queries, keys.transpose(-2, -1))
        # 
        sequence_mask = torch.triu(torch.ones(queries.size(-2), keys.size(-2)), diagonal=1).bool().to("cuda")
        attention.masked_fill_(sequence_mask, float("-inf"))
        attention = attention/(self.head_dim**0.5)
        attention = torch.softmax(attention, dim =-1)

        return self.merge_heads(torch.matmul(attention, values))
    
    def forward(self, x):
        queries = self.split_heads(self.QW(x))
        keys = self.split_heads(self.KW(x))
        values = self.split_heads(self.VW(x))

        concatened_values = self.apply_attention(queries, keys, values)
        output = self.WO(concatened_values)

        return output
        
   
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
   
        self.hidden_size=hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias  = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_params()

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        variance = input.var(dim=-1, unbiased=False, keepdim=True)
        outputs = (input - mean) / torch.sqrt(variance + self.eps)
        outputs = outputs*self.weight + self.bias
        return outputs 
    
    def reset_params(self, ):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class Block(nn.Module):
    def __init__(self,head_dim, mlp_hidden_size, num_heads, sequence_len):
        super(Block, self).__init__()
        self.head_dim = head_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads
        self.sequence_len = sequence_len

        self.hidden_size = num_heads * head_dim

        self.attention = MHA(head_dim, num_heads, sequence_len)
        self.layer_norm1 = LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_size, self.mlp_hidden_size), 
                                 nn.GELU(),
                                 nn.Linear(self.mlp_hidden_size, self.hidden_size)
                                 )
        self.layer_norm2 = LayerNorm(self.hidden_size)
    
    def forward(self, x):
        attention_output = self.attention(x)
        attention_output = self.layer_norm1(x + attention_output) # investigate more 
        mlp_output = self.mlp(attention_output)
        output = self.layer_norm2(attention_output + mlp_output)

        return output


class GPT(nn.Module):
    def __init__(self, vocabulary_size=50000, 
                 embedding_size=768,  
                 sequence_len=512, 
                 num_layers=4, 
                 num_heads=2):
        super(GPT, self).__init__()

        self.embedding = GPTEmbedding(vocabulary_size, embedding_size, sequence_len)
        self.num_layers = num_layers
        self.head_dim = embedding_size // num_heads
        self.layers = nn.ModuleList([Block(self.head_dim, 4*embedding_size, num_heads, sequence_len) for i in range(num_layers)])
        self.classifier = nn.Linear(embedding_size, vocabulary_size)

    
    def forward(self, tokens):
        positions = torch.arange(tokens.size(1)).unsqueeze(0).repeat(tokens.size(0), 1).to("cuda")
        embeddings = self.embedding(tokens, positions).to("cuda")
        for layer in self.layers:
            embeddings = layer(embeddings)
        
        logits = self.classifier(embeddings)
        log_probs = torch.log_softmax(logits, dim=-1)
        # B x S x V 

        return log_probs
    
    def loss(self, log_probs, target):
        # Input -> [a,b,c,d,e]
        # Target -> [b,c,d,e] B X S x 1
        # B x S x V
        logp_q = log_probs.gather(2, target.unsqueeze(2).long()) / target.size(0)
        nll = -torch.sum(logp_q)
        return nll 


        

