import torch
import torch.nn as nn
import tiktoken
tokenizer=tiktoken.get_encoding("gpt2")

#Specifying  GPT with Dict

GPT_Config_124M= {

    "vocab_size" : 50257, #Vocabulary size for GPT2
    "context_length" : 1024, # Context Length
    "emb_dim":768, # Embedding Dimension
    "n_heads":12, # 12 attention heads
    "n_layers":12, # Number of layers
    "drop_rate":0.1, #Dropout rate
    "qkv_bias": False #Query,key,value bias

}

# Now implementing a class to use in GPT model later

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps=1e-5 # to avoid division by zero errors
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x=(x-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm_x+self.shift
    


# Now thwe GELU class
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1+ torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))* 
                                        (x+0.044715*torch.pow(x,3))))


class Feedforward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
                                  GELU(),
                                  nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]))
        
    def forward(self,x):
        return self.layers(x)


    

# importing the MultiAttentionHead class 
from MAH import MultiAttentionHead

#Defining the transformer block
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att=MultiAttentionHead(
            dim_in=cfg["emb_dim"],
            dim_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff=Feedforward(cfg)
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg["emb_dim"])
        self.drop_shortcut=nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
    
        shortcut=x
        # shortcut for the attention module
        x=self.norm1(x)
        x=self.att(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        # shortcut for the feedforward module
        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        return x



#Architecture
# Actual working GPT model


class GPT_Model(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # definiing token emb and pos emb
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])

        #dropout rate
        self.drop_emb=nn.Dropout(cfg["drop_rate"])

        #now the transformer block with MHA,ReLU,GELU,etc

        self.trf_block=nn.Sequential(
            *[TransformerBlock(cfg)
            for _ in range(cfg["n_layers"])])
        

        # mpw the layer normalization
        self.layer_norm= LayerNorm(cfg["emb_dim"])

        # output projector to vocab
        self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)


    def forward(self,in_idx):
       # getting batch size and seq length
     batch_size,seq_len= in_idx.shape # eg [2,8,768] will assign 2 batch size and 8 tokens as seq_len
     # tok emb and pos emb
     tok_emb=self.tok_emb(in_idx)            # allows to train oc CPU or GPU wherever is the data sitting on
     pos_emb=self.pos_emb(torch.arange(seq_len,device=in_idx.device))

     # now emb vector
     x=tok_emb+pos_emb
     #applying the blocks
     x=self.drop_emb(x)
     x=self.trf_block(x)
     x=self.layer_norm(x)

     # now the logits
     logits=self.out_head(x)

     return logits



# defining generate text function
def generate_text_simple(model,idx,max_new_tokens,context_size):
       for _ in range (max_new_tokens):
              idx_cond=idx[:,-context_size:] # we only care for the context size, first extra half is ignored (if bigger)
              with torch.no_grad():
                     logits=model(idx_cond)

              logits=logits[:,-1,:] # select only the recent token
              probas=torch.softmax(logits,dim=-1) # probability with the softmax applied for the recent most token vector
              idx_next=torch.argmax(probas,dim=-1,keepdim=True)# the idx_next is the new token with max val
              idx=torch.cat((idx,idx_next),dim=-1)# add the new token to the sequence

       return idx


#testing the function
torch.manual_seed(123)

model=GPT_Model(GPT_Config_124M)
start_context="1+2"
encoded=tokenizer.encode(start_context) # already defined tiktoken gpt2
print("encoded: ",encoded)
# need to add batch dimension
encoded_tensor=torch.tensor(encoded).unsqueeze(0)

print("encoded tensor: ",encoded_tensor, "Shape: ",encoded_tensor.shape)

# now we will call the model but first put it into eval() mode to disable dropout etc which are only needed
# during training

model.eval()
output=generate_text_simple(model=model,idx=encoded_tensor,max_new_tokens=6,
                            context_size=GPT_Config_124M["context_length"])

print("Output: ", output) # the model has generated token ids, we need to convert using decode
print("Output_length: ", len(output[0]) )
decoded_text=tokenizer.decode(output.squeeze(0).tolist())
print(decoded_text) # first ever output text