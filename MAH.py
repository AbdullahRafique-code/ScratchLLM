import torch.nn as nn
import torch




class MultiAttentionHead(nn.Module):
    def __init__(self,dim_in,dim_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()#nn.Module
        assert dim_out % num_heads==0, "dimension output must be divisble by num of heads"
        self.dim_out=dim_out
        self.num_heads=num_heads
        self.head_dim=dim_out//num_heads
        self.w_query=nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.w_key=nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.w_value=nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.out_proj=nn.Linear(dim_out,dim_out)    
        self.dropout=nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )

    def forward(self,x):
        b,num_tokens,dim_in=x.shape
        keys=self.w_key(x)
        queries=self.w_query(x)
        values=self.w_value(x)

        #Now we change the shape of keys, values and quereis to accomodate all heads and head dim

        keys=keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values=values.view(b,num_tokens,self.num_heads,self.head_dim)
        queries=queries.view(b,num_tokens,self.num_heads,self.head_dim)
    

        #No of heads is like batch dimension, we need to transpose to shift num of tokens (context) 
        # with batch (num of heads)

        keys = keys.transpose(1,2)
        values= values.transpose(1,2)
        queries= queries.transpose(1,2)


        #calculate
        attention_scores=queries@keys.transpose(2,3) 
        mask_bool=self.mask.bool()[:num_tokens,:num_tokens]

        attention_scores.masked_fill_(mask_bool,-torch.inf)

        attention_weights=torch.softmax(attention_scores/keys.shape[-1]**0.5,dim=-1)

        attention_weights=self.dropout(attention_weights)

        context_vector=(attention_weights@values).transpose(1,2)

        context_vector=context_vector.contiguous().view(b,num_tokens,self.dim_out)

        context_vector=self.out_proj(context_vector)


        return context_vector
    




