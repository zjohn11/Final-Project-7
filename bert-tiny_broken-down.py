import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# loading the bert-tiny model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained("./bert-tiny_sentiment-analysis", num_labels=2)  # Binary classification

# EMBEDDINGS LAYER
class EmbeddingLayer:
    def __init__(self, embedding_matrix, positional_embeddings, token_type_embeddings, layerNorm_weight, layerNorm_bias):
        super().__init__()
        self.embeddings = embedding_matrix
        self.positional = positional_embeddings
        self.token_type = token_type_embeddings
        self.layernorm_weight = layerNorm_weight
        self.layernorm_bias = layerNorm_bias

    def forward(self,x):
        seq_len = len(x)
        # WORD EMBEDDINGS
        word_embeddings = self.embeddings[x]

        # POSITIONAL EMBEDDINGS
        positional_embeddings = self.positional[:seq_len, :]

        # SEGMENT EMBEDDINGS
        segment_embeddings = [0] * seq_len
        segment_embeddings = self.token_type[segment_embeddings,:]
        segment_embeddings = segment_embeddings[:seq_len, :]

        # EMBEDDINGS SUM
        final_embeddings = word_embeddings + segment_embeddings + positional_embeddings

        # LAYER NORMALIZATION
        for index in range(final_embeddings.shape[0]):
            token_mat = final_embeddings[index,:]

            # Compute the mean and standard deviation along the embedding dimension (axis 0)
            mean = np.mean(token_mat)
            std = np.std(token_mat)

            # Normalize the token embedding (subtract mean and divide by std)
            normalized_token_mat = (token_mat - mean) / (std + 1e-12)

            final_embeddings[index,:] = layernorm_weight * normalized_token_mat + layernorm_bias
        
        return final_embeddings
    
# TRANSFORMER LAYER
class TransformerLayer:
    def __init__(self, query_weight, query_bias, key_weight, key_bias, value_weight, value_bias, attention_dense_weight, attention_dense_bias, attention_layerNorm_weight, 
                 attention_layerNorm_bias,intermediate_dense_weight, intermediate_dense_bias, output_dense_weight, output_dense_bias, output_layerNorm_weight, output_layerNorm_bias):
        super().__init__()
        # FIRST ATTENTION HEAD MATRICES #
        self.q_weight = query_weight
        self.q_bias = query_bias
        self.k_weight = key_weight
        self.k_bias = key_bias
        self.v_weight = value_weight
        self.v_bias = value_bias
        # ATTENTION DENSE #
        self.attention_dense_weight = attention_dense_weight
        self.attention_dense_bias = attention_dense_bias
        # ATTENTION LAYER NORMALIZATION #
        self.attention_layerNorm_weight = attention_layerNorm_weight
        self.attention_layerNorm_bias = attention_layerNorm_bias
        # FEEDFORWARD NETWORK PARAMETERS #
        self.intermediate_dense_weight = intermediate_dense_weight
        self.intermediate_dense_bias = intermediate_dense_bias
        self.output_dense_weight = output_dense_weight
        self.output_dense_bias = output_dense_bias
        self.output_layerNorm_weight = output_layerNorm_weight
        self.output_layerNorm_bias = output_layerNorm_bias
        # NUMBER OF ATTENTION HEADS #
        self.heads = 2
        self.d_k = 64 # hidden size / number of attention heads
    
    def transformer_forward(self,x):
        ## Calculating Query, Key, and Value from embedding output and pretrained model weights and biases
        seq_len = len(x)
        # FIRST ATTENTION HEAD #
        q1 = np.matmul(x,np.transpose(self.q_weight)) + self.q_bias
        k1 = np.matmul(x,np.transpose(self.k_weight)) + self.k_bias
        v1 = np.matmul(x,np.transpose(self.v_weight)) + self.v_bias
        # RESHAPING Q,K,V #
        q1 = q1.reshape(seq_len, self.heads, self.d_k)  # Reshape into (seq_len, heads, d_k)
        k1 = k1.reshape(seq_len, self.heads, self.d_k)  # Reshape into (seq_len, heads, d_k)
        v1 = v1.reshape(seq_len, self.heads, self.d_k)  # Reshape into (seq_len, heads, d_k)
        # TRANSPOSE OF Q,K,V #
        q1 = np.transpose(q1, (1, 0, 2))  # (heads, seq_len, d_k)
        k1 = np.transpose(k1, (1, 0, 2))  # (heads, seq_len, d_k)
        v1 = np.transpose(v1, (1, 0, 2))  # (heads, seq_len, d_k)
        # ATTENTION SCORES #
        attention_score1 = np.matmul(q1,np.transpose(k1,(0, 2, 1)))
        # SCALED ATTENTION SCORES #
        scaled_score1 = attention_score1/(np.sqrt(self.d_k))
        # SOFTMAX #
        exp_x1 = np.exp(scaled_score1 - np.max(scaled_score1, axis=-1, keepdims=True))  # Stabilize with max subtraction
        sum_exp_x1 = np.sum(exp_x1, axis=-1, keepdims=True)  # Sum along the last axis (seq_len)
        softmax1 = exp_x1 / sum_exp_x1  # Normalize
        # CONTEXT #
        context1 = np.matmul(softmax1,v1)
        context1 = context1.transpose(1,0,2).reshape(seq_len, self.heads * self.d_k)

        # ATTENTION LINEAR PROJECTION # 
        output_projected = np.matmul(context1,np.transpose(self.attention_dense_weight)) + self.attention_dense_bias
        # RESIDUAL ADDITON #
        output_projected = output_projected + x
        # ATTENTION LAYER NORMALIZATION #
        for index in range(output_projected.shape[0]):
            token_mat = output_projected[index,:]

            # Compute the mean and standard deviation along the embedding dimension (axis 0)
            mean = np.mean(token_mat)
            std = np.std(token_mat)

            # Normalize the token embedding (subtract mean and divide by std)
            normalized_token_mat = (token_mat - mean) / (std + 1e-12)

            output_projected[index,:] = self.attention_layerNorm_weight * normalized_token_mat + self.attention_layerNorm_bias

        # FEEDFORWARD NETWORK #
        ffn_layer1 = np.matmul(output_projected,np.transpose(self.intermediate_dense_weight)) + self.intermediate_dense_bias
        ffn_layer1 = 0.5 * ffn_layer1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (ffn_layer1 + 0.044715 * np.power(ffn_layer1, 3)))) #GELU
        ffn_layer2 = np.matmul(ffn_layer1,np.transpose(self.output_dense_weight)) + self.output_dense_bias
        # RESIDUAL ADDITION # 
        ffn_layer2 = ffn_layer2 + output_projected
        # FFN LAYER NORMALIZATION #
        for index in range(ffn_layer2.shape[0]):
            token_mat = ffn_layer2[index,:]

            # Compute the mean and standard deviation along the embedding dimension (axis 0)
            mean = np.mean(token_mat)
            std = np.std(token_mat)

            # Normalize the token embedding (subtract mean and divide by std)
            normalized_token_mat = (token_mat - mean) / (std + 1e-12)

            ffn_layer2[index,:] = self.output_layerNorm_weight * normalized_token_mat + self.output_layerNorm_bias
        
        return ffn_layer2

# PREDICTION HEAD
class PredictionHead:
    def __init__(self, pooler_weight, pooler_bias, classifier_weight, classifier_bias):
        super().__init__()
        # POOLER
        self.pooler_w = pooler_weight
        self.pooler_b = pooler_bias
        # CLASSIFIER
        self.classifier_w = classifier_weight
        self.classifier_b = classifier_bias
    
    def forward(self,x):
        # GETTING [CLS] TOKEN
        cls_tok = x[0]
        # PROJECTING OUTPUT FROM TRANSFORMER
        transformer_pooled = np.matmul(cls_tok, np.transpose(self.pooler_w)) + pooler_bias
        # TANH ACTIVATION FUNCTION
        transformer_pooled = np.tanh(transformer_pooled)
        # CLASSIFIER LINEAR LAYER
        classified_output = np.matmul(transformer_pooled, np.transpose(self.classifier_w)) + classifier_bias
        # SOFTMAX #
        exp_x1 = np.exp(classified_output - np.max(classified_output, axis=-1, keepdims=True))  # Stabilize with max subtraction
        sum_exp_x1 = np.sum(exp_x1, axis=-1, keepdims=True)  # Sum along the last axis (seq_len)
        softmax_output = exp_x1 / sum_exp_x1  # Normalize

        return softmax_output
#--------------------------------------------------------------------------------------------------------------------#

# MODEL VOCABULARY
vocabulary_dict = tokenizer.get_vocab()
# MODEL EMBEDDING MATRIX
embedding_matrix = model.bert.embeddings.word_embeddings.weight.detach().numpy()
# MODEL POSITION EMBEDDINGS
position_embeddings = model.bert.embeddings.position_embeddings.weight.detach().numpy()
# MODEL TOKEN TYPE EMBEDDINGS
token_type_embeddings = model.bert.embeddings.token_type_embeddings.weight.detach().numpy()
# MODEL LAYER NORMALIZATION EMBEDDINGS
layernorm_weight = model.bert.embeddings.LayerNorm.weight.detach().numpy()
layernorm_bias = model.bert.embeddings.LayerNorm.bias.detach().numpy()

# ---------------LAYER 1--------------------------------------------------------------------------------------------#
# MODEL QUERY, KEY, AND VALUE MATRICES
layer1_query_weight = model.bert.encoder.layer[0].attention.self.query.weight.detach().numpy()
layer1_query_bias = model.bert.encoder.layer[0].attention.self.query.bias.detach().numpy()
layer1_key_weight = model.bert.encoder.layer[0].attention.self.key.weight.detach().numpy()
layer1_key_bias = model.bert.encoder.layer[0].attention.self.key.bias.detach().numpy()
layer1_value_weight = model.bert.encoder.layer[0].attention.self.value.weight.detach().numpy()
layer1_value_bias = model.bert.encoder.layer[0].attention.self.value.bias.detach().numpy()
# DENSE LAYER WEIGHTS AND BIASES
layer1_attention_dense_weight = model.bert.encoder.layer[0].attention.output.dense.weight.detach().numpy()
layer1_attention_dense_bias = model.bert.encoder.layer[0].attention.output.dense.bias.detach().numpy()
layer1_intermediate_dense_weight = model.bert.encoder.layer[0].intermediate.dense.weight.detach().numpy()
layer1_intermediate_dense_bias = model.bert.encoder.layer[0].intermediate.dense.bias.detach().numpy()
layer1_output_dense_weight = model.bert.encoder.layer[0].output.dense.weight.detach().numpy()
layer1_output_dense_bias = model.bert.encoder.layer[0].output.dense.bias.detach().numpy()
# LAYER NORMALIZATION PARAMETERS
layer1_attention_layerNorm_weight = model.bert.encoder.layer[0].attention.output.LayerNorm.weight.detach().numpy()
layer1_attention_layerNorm_bias = model.bert.encoder.layer[0].attention.output.LayerNorm.bias.detach().numpy()
layer1_output_layerNorm_weight = model.bert.encoder.layer[0].output.LayerNorm.weight.detach().numpy()
layer1_output_layerNorm_bias = model.bert.encoder.layer[0].output.LayerNorm.bias.detach().numpy()

# ---------------LAYER 2--------------------------------------------------------------------------------------------#
# MODEL QUERY, KEY, AND VALUE MATRICES
layer2_query_weight = model.bert.encoder.layer[1].attention.self.query.weight.detach().numpy()
layer2_query_bias = model.bert.encoder.layer[1].attention.self.query.bias.detach().numpy()
layer2_key_weight = model.bert.encoder.layer[1].attention.self.key.weight.detach().numpy()
layer2_key_bias = model.bert.encoder.layer[1].attention.self.key.bias.detach().numpy()
layer2_value_weight = model.bert.encoder.layer[1].attention.self.value.weight.detach().numpy()
layer2_value_bias = model.bert.encoder.layer[1].attention.self.value.bias.detach().numpy()
# DENSE LAYER WEIGHTS AND BIASES
layer2_attention_dense_weight = model.bert.encoder.layer[1].attention.output.dense.weight.detach().numpy()
layer2_attention_dense_bias = model.bert.encoder.layer[1].attention.output.dense.bias.detach().numpy()
layer2_intermediate_dense_weight = model.bert.encoder.layer[1].intermediate.dense.weight.detach().numpy()
layer2_intermediate_dense_bias = model.bert.encoder.layer[1].intermediate.dense.bias.detach().numpy()
layer2_output_dense_weight = model.bert.encoder.layer[1].output.dense.weight.detach().numpy()
layer2_output_dense_bias = model.bert.encoder.layer[1].output.dense.bias.detach().numpy()
# LAYER NORMALIZATION PARAMETERS
layer2_attention_layerNorm_weight = model.bert.encoder.layer[1].attention.output.LayerNorm.weight.detach().numpy()
layer2_attention_layerNorm_bias = model.bert.encoder.layer[1].attention.output.LayerNorm.bias.detach().numpy()
layer2_output_layerNorm_weight = model.bert.encoder.layer[1].output.LayerNorm.weight.detach().numpy()
layer2_output_layerNorm_bias = model.bert.encoder.layer[1].output.LayerNorm.bias.detach().numpy()

# ---------------CLASSIFIER HEAD--------------------------------------------------------------------------------------------#
pooler_weight = model.bert.pooler.dense.weight.detach().numpy()
pooler_bias = model.bert.pooler.dense.bias.detach().numpy()
classifier_weight = model.classifier.weight.detach().numpy()
classifier_bias = model.classifier.bias.detach().numpy()

# INPUT TEXT
text = ["[CLS]", "i", "liked","some", "parts","and", "hated", "other","parts", "of", "the", "movie" ".", "[SEP]"]

# VOCABULARY LOOK-UP TABLE
input_ids = [vocabulary_dict.get(token, vocabulary_dict["[UNK]"]) for token in text]

# EMBEDDING LAYER
embedding_layer = EmbeddingLayer(embedding_matrix, position_embeddings, token_type_embeddings, layernorm_weight, layernorm_bias)

# TRANSFORMER LAYER 1
transformer_layer1 = TransformerLayer(layer1_query_weight, layer1_query_bias, layer1_key_weight, layer1_key_bias,
                                      layer1_value_weight, layer1_value_bias, layer1_attention_dense_weight, layer1_attention_dense_bias,
                                      layer1_attention_layerNorm_weight, layer1_attention_layerNorm_bias,layer1_intermediate_dense_weight,
                                      layer1_intermediate_dense_bias,layer1_output_dense_weight,layer1_output_dense_bias,
                                      layer1_output_layerNorm_weight,layer1_output_layerNorm_bias)

# TRANSFORMER LAYER 2
transformer_layer2 = TransformerLayer(layer2_query_weight, layer2_query_bias, layer2_key_weight, layer2_key_bias,
                                      layer2_value_weight, layer2_value_bias, layer2_attention_dense_weight, layer2_attention_dense_bias,
                                      layer2_attention_layerNorm_weight, layer2_attention_layerNorm_bias,layer2_intermediate_dense_weight,
                                      layer2_intermediate_dense_bias,layer2_output_dense_weight,layer2_output_dense_bias,
                                      layer2_output_layerNorm_weight,layer2_output_layerNorm_bias)

# PREDICTION HEAD LAYER
pred_head = PredictionHead(pooler_weight, pooler_bias, classifier_weight, classifier_bias)

# OUTPUT FROM EMBEDDING LAYER
final_embeddings = embedding_layer.forward(input_ids)

# OUTPUT FROM TRANSFORMER LAYER 1
transformer_output1 = transformer_layer1.transformer_forward(final_embeddings)

# OUTPUT FROM TRANSFORMER LAYER 2
transformer_output2 = transformer_layer2.transformer_forward(transformer_output1)

# OUTPUT FROM PREDICTION HEAD
pred_output = pred_head.forward(transformer_output2)

# PRINTING SOFTMAX OUTPUT
print(pred_output)

# GETTING SENTIMENT
sentiment = np.argmax(pred_output)

if sentiment == 0:
    print("Negative")
else:
    print("Positive")
