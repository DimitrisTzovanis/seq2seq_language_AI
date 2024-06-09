import numpy as np

import typing
from typing import Any, Tuple

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
import tensorflow_text as tf_text
import pandas as pd


#@title
class ShapeChecker():
  def __init__(self):
    # Keep a cache of every axis-name seen
    self.shapes = {}

  def __call__(self, tensor, names, broadcast=False):
    if not tf.executing_eagerly():
      return

    parsed = einops.parse_shape(tensor, names)

    for name, new_dim in parsed.items():
      old_dim = self.shapes.get(name, None)
      
      if (broadcast and new_dim == 1):
        continue

      if old_dim is None:
        # If the axis name is new, add its length to the cache.
        self.shapes[name] = new_dim
        continue

      if new_dim != old_dim:
        raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                         f"    found: {new_dim}\n"
                         f"    expected: {old_dim}\n")




def load_data(csv_path):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_path)
    
    # Ensure that columns are strings and handle potential missing values
    context = data['user'].astype(str).fillna('missing').to_numpy()
    target = data['response'].astype(str).fillna('missing').to_numpy()

    return target, context

# Usage
csv_path = 'conversations.csv'  # Update this path as necessary
target_raw, context_raw = load_data(csv_path)
print(context_raw[-1])


print(target_raw[-1])



BUFFER_SIZE = len(context_raw)
BATCH_SIZE = 64

is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

train_raw = (
    tf.data.Dataset
    .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))
val_raw = (
    tf.data.Dataset
    .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))


for example_context_strings, example_target_strings in train_raw.take(1):
  print(example_context_strings[:5])
  print()
  print(example_target_strings[:5])
  break

example_text = tf.constant('Τι κανει;')

print(example_text.numpy())
print(tf_text.normalize_utf8(example_text, 'NFKD').numpy())


def tf_lower_and_split_punct(text):
  # Normalize and split accented characters using NFKD.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, A to Z, Greek alphabet, and select punctuation.
    # The regex pattern includes the Unicode range for Greek characters.
    text = tf.strings.regex_replace(text, '[^ a-zα-ωά-ώΑ-ΩΆ-Ώ.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.;?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.regex_replace(text, 'ά', 'α')
    text = tf.strings.regex_replace(text, 'έ', 'ε')
    text = tf.strings.regex_replace(text, 'ή', 'η')
    
    text = tf.strings.strip(text)

    # Surround the text with start and end tokens.
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


print(example_text.numpy().decode())
print(tf_lower_and_split_punct(example_text).numpy().decode())


max_vocab_size = 30000


token_file_path = 'pt_vocab.txt'
text_ds = tf.data.Dataset.from_tensor_slices(context_raw)
# Read the file and split into tokens
with open(token_file_path, 'r', encoding='utf-8') as file:
    custom_tokens = [line.strip() for line in file]

# Add custom tokens to the dataset
custom_token_ds = tf.data.Dataset.from_tensor_slices(custom_tokens)
target_ds = train_raw.map(lambda context, target: context)


combined_ds = target_ds.concatenate(custom_token_ds)


# Define your TextVectorization layer, if not already defined
context_text_processor = tf.keras.layers.TextVectorization(
    standardize=None,  # Consider setting to None to avoid altering tokens like '##example'
    max_tokens=None,   # Adjust or remove limit as needed
    output_mode='int',
    ragged=True
)

# Adapt the TextVectorization layer
# Try without batching, or adjust the batch size
combined_ds = text_ds.concatenate(custom_token_ds).batch(32)  # Adjust batch size here if needed
context_text_processor.adapt(combined_ds)



# Here are the first 10 words from the vocabulary:
context_text_processor.get_vocabulary()[:100]

example_tokens = context_text_processor(example_context_strings)
example_tokens[:3, :]

def check_unknowns(text, text_processor):
    # Normalize and preprocess the text to match the expected input format
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-zα-ωά-ώΑ-ΩΆ-Ώ.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)

    # Split the processed text into tokens
    tokens = tf.strings.split(text, ' ')

    # Flatten the tokens to ensure we have a 1D list of tokens
    tokens = tokens.flat_values if isinstance(tokens, tf.RaggedTensor) else tokens

    # Get the vocabulary and convert it to a Python set for fast lookup
    vocab_set = set(text_processor.get_vocabulary())

    # Convert each token to string and check if it's not in the vocabulary set
    unknown_tokens = [token.numpy().decode('utf-8') for token in tokens if token.numpy().decode('utf-8') not in vocab_set]

    return unknown_tokens



# Example usage
sample_texts = tf.constant(["Η τυπισα ειναι σαν γαμω την μανα σου Ειναι αυτουνου Κριμα που δεν τον καλεσα στο τετοιο Για 3; Θ το δεχτει Με τον γιαννη; Αυριο στις 10 θα παμε; Εχω φτασει Βαρεθηκα ορθοστασια Αν πηγες Και εσυ βαριοσουνα να πας Φαση σε καμια ωρα Σε ενα 10λεπτο ξεκινω Βεζω βενζυνα 20 μαξ Για βγες μπροστα Ειμαι φουλ κοντα"])
unknown_words = check_unknowns(context_raw, context_text_processor)

# Path to your custom tokens file
tokens_file_path = 'pt_vocab.txt'

# Append unknown words to the file
with open(tokens_file_path, 'a', encoding='utf-8') as file:
    for word in unknown_words:
        file.write(word + '\n')




token_file_path = 'en_vocab.txt'
text_ds = tf.data.Dataset.from_tensor_slices(target_raw)
# Read the file and split into tokens
with open(token_file_path, 'r', encoding='utf-8') as file:
    custom_tokens = [line.strip() for line in file]

# Add custom tokens to the dataset
custom_token_ds = tf.data.Dataset.from_tensor_slices(custom_tokens)
target_ds = train_raw.map(lambda context, target: target)


combined_ds = target_ds.concatenate(custom_token_ds)


# Define your TextVectorization layer, if not already defined
target_text_processor = tf.keras.layers.TextVectorization(
    standardize=None,  # Consider setting to None to avoid altering tokens like '##example'
    max_tokens=None,   # Adjust or remove limit as needed
    output_mode='int',
    ragged=True
)

# Adapt the TextVectorization layer
# Try without batching, or adjust the batch size
combined_ds = text_ds.concatenate(custom_token_ds).batch(32)  # Adjust batch size here if needed
target_text_processor.adapt(combined_ds)




def check_unknowns(text, text_processor):
    # Normalize and preprocess the text to match the expected input format
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-zα-ωά-ώΑ-ΩΆ-Ώ.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)

    # Split the processed text into tokens
    tokens = tf.strings.split(text, ' ')

    # Flatten the tokens to ensure we have a 1D list of tokens
    tokens = tokens.flat_values if isinstance(tokens, tf.RaggedTensor) else tokens

    # Get the vocabulary and convert it to a Python set for fast lookup
    vocab_set = set(text_processor.get_vocabulary())

    # Convert each token to string and check if it's not in the vocabulary set
    unknown_tokens = [token.numpy().decode('utf-8') for token in tokens if token.numpy().decode('utf-8') not in vocab_set]

    return unknown_tokens



# Example usage
sample_texts = tf.constant(["Η τυπισα ειναι σαν γαμω την μανα σου Ειναι αυτουνου Κριμα που δεν τον καλεσα στο τετοιο Για 3; Θ το δεχτει Με τον γιαννη; Αυριο στις 10 θα παμε; Εχω φτασει Βαρεθηκα ορθοστασια Αν πηγες Και εσυ βαριοσουνα να πας Φαση σε καμια ωρα Σε ενα 10λεπτο ξεκινω Βεζω βενζυνα 20 μαξ Για βγες μπροστα Ειμαι φουλ κοντα"])
unknown_words = check_unknowns(target_raw, target_text_processor)

# Path to your custom tokens file
tokens_file_path = 'en_vocab.txt'

# Append unknown words to the file
with open(tokens_file_path, 'a', encoding='utf-8') as file:
    for word in unknown_words:
        file.write(word + '\n')





# Example text to see how it's tokenized
example_text = tf.constant(["Η τυπισα ειναι σαν γαμω την μανα σου Ειναι αυτουνου Κριμα που δεν τον καλεσα στο τετοιο Για 3; Θ το δεχτει Με τον γιαννη; Αυριο στις 10 θα παμε; Εχω φτασει Βαρεθηκα ορθοστασια Αν πηγες Και εσυ βαριοσουνα να πας Φαση σε καμια ωρα Σε ενα 10λεπτο ξεκινω Βεζω βενζυνα 20 μαξ Για βγες μπροστα Ειμαι φουλ κοντα"])


# Process the text through the text processor
processed_text = target_text_processor(example_text)

# Print the processed text
print("Processed text:", processed_text)

# To see the actual words corresponding to the token IDs:
vocab = target_text_processor.get_vocabulary()
print("Vocabulary sample:", vocab[:100])  # Adjust indexing to see more or fewer vocab items

# Map token IDs back to words
words = [vocab[id] for id in processed_text.numpy()[0]]
print("Tokenized words:", words)






context_vocab = np.array(context_text_processor.get_vocabulary())
tokens = context_vocab[example_tokens[0].numpy()]
' '.join(tokens)


plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens.to_tensor())
plt.title('Token IDs')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens.to_tensor() != 0)
plt.title('Mask')


def process_text(context, target):
  context = context_text_processor(context).to_tensor()
  target = target_text_processor(target)
  targ_in = target[:,:-1].to_tensor()
  targ_out = target[:,1:].to_tensor()
  return (context, targ_in), targ_out


train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)


for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
  print(ex_context_tok[0, :10].numpy()) 
  print()
  print(ex_tar_in[0, :10].numpy()) 
  print(ex_tar_out[0, :10].numpy())


UNITS = 256


print(context_text_processor.get_vocabulary())


# Example Greek phrase
test_phrase = 'Καλημέρα'  # Good morning
processed_phrase = tf_lower_and_split_punct(tf.constant(test_phrase))
print(processed_phrase.numpy().decode())






class Encoder(tf.keras.layers.Layer):
  def __init__(self, text_processor, units):
    super(Encoder, self).__init__()
    self.text_processor = text_processor
    self.vocab_size = text_processor.vocabulary_size()
    self.units = units
    
    # The embedding layer converts tokens to vectors
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, units,
                                               mask_zero=True)

    # The RNN layer processes those vectors sequentially.
    self.rnn = tf.keras.layers.Bidirectional(
        merge_mode='sum',
        layer=tf.keras.layers.GRU(units,
                            # Return the sequence and state
                            return_sequences=True,
                            recurrent_initializer='glorot_uniform'))

  def call(self, x):
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch s')

    # 2. The embedding layer looks up the embedding vector for each token.
    x = self.embedding(x)
    shape_checker(x, 'batch s units')

    # 3. The GRU processes the sequence of embeddings.
    x = self.rnn(x)
    shape_checker(x, 'batch s units')

    # 4. Returns the new sequence of embeddings.
    return x

  def convert_input(self, texts):
    texts = tf.convert_to_tensor(texts)
    if len(texts.shape) == 0:
      texts = tf.convert_to_tensor(texts)[tf.newaxis]
    context = self.text_processor(texts).to_tensor()
    context = self(context)
    return context




# Encode the input sequence.
encoder = Encoder(context_text_processor, UNITS)
ex_context = encoder(ex_context_tok)

print(f'Context tokens, shape (batch, s): {ex_context_tok.shape}')
print(f'Encoder output, shape (batch, s, units): {ex_context.shape}')




class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):
    shape_checker = ShapeChecker()
 
    shape_checker(x, 'batch t units')
    shape_checker(context, 'batch s units')

    attn_output, attn_scores = self.mha(
        query=x,
        value=context,
        return_attention_scores=True)
    
    shape_checker(x, 'batch t units')
    shape_checker(attn_scores, 'batch heads t s')
    
    # Cache the attention scores for plotting later.
    attn_scores = tf.reduce_mean(attn_scores, axis=1)
    shape_checker(attn_scores, 'batch t s')
    self.last_attention_weights = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x




attention_layer = CrossAttention(UNITS)

# Attend to the encoded tokens
embed = tf.keras.layers.Embedding(target_text_processor.vocabulary_size(),
                                  output_dim=UNITS, mask_zero=True)
ex_tar_embed = embed(ex_tar_in)

result = attention_layer(ex_tar_embed, ex_context)

print(f'Context sequence, shape (batch, s, units): {ex_context.shape}')
print(f'Target sequence, shape (batch, t, units): {ex_tar_embed.shape}')
print(f'Attention result, shape (batch, t, units): {result.shape}')
print(f'Attention weights, shape (batch, t, s):    {attention_layer.last_attention_weights.shape}')


attention_layer.last_attention_weights[0].numpy().sum(axis=-1)




attention_weights = attention_layer.last_attention_weights
mask=(ex_context_tok != 0).numpy()

plt.subplot(1, 2, 1)
plt.pcolormesh(mask*attention_weights[:, 0, :])
plt.title('Attention weights')

plt.subplot(1, 2, 2)
plt.pcolormesh(mask)
plt.title('Mask');




class Decoder(tf.keras.layers.Layer):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, text_processor, units):
    super(Decoder, self).__init__()
    self.text_processor = text_processor
    self.vocab_size = text_processor.vocabulary_size()
    self.word_to_id = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]')
    self.id_to_word = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]',
        invert=True)
    self.start_token = self.word_to_id('[START]')
    self.end_token = self.word_to_id('[END]')

    self.units = units


    # 1. The embedding layer converts token IDs to vectors
    self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                               units, mask_zero=True)

    # 2. The RNN keeps track of what's been generated so far.
    self.rnn = tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # 3. The RNN output will be the query for the attention layer.
    self.attention = CrossAttention(units)

    # 4. This fully connected layer produces the logits for each
    # output token.
    self.output_layer = tf.keras.layers.Dense(self.vocab_size)


@Decoder.add_method
def call(self,
         context, x,
         state=None,
         return_state=False):  
  shape_checker = ShapeChecker()
  shape_checker(x, 'batch t')
  shape_checker(context, 'batch s units')

  # 1. Lookup the embeddings
  x = self.embedding(x)
  shape_checker(x, 'batch t units')

  # 2. Process the target sequence.
  x, state = self.rnn(x, initial_state=state)
  shape_checker(x, 'batch t units')

  # 3. Use the RNN output as the query for the attention over the context.
  x = self.attention(x, context)
  self.last_attention_weights = self.attention.last_attention_weights
  shape_checker(x, 'batch t units')
  shape_checker(self.last_attention_weights, 'batch t s')

  # Step 4. Generate logit predictions for the next token.
  logits = self.output_layer(x)
  shape_checker(logits, 'batch t target_vocab_size')

  if return_state:
    return logits, state
  else:
    return logits




decoder = Decoder(target_text_processor, UNITS)


logits = decoder(ex_context, ex_tar_in)

print(f'encoder output shape: (batch, s, units) {ex_context.shape}')
print(f'input target tokens shape: (batch, t) {ex_tar_in.shape}')
print(f'logits shape shape: (batch, target_vocabulary_size) {logits.shape}')



@Decoder.add_method
def get_initial_state(self, context):
  batch_size = tf.shape(context)[0]
  start_tokens = tf.fill([batch_size, 1], self.start_token)
  done = tf.zeros([batch_size, 1], dtype=tf.bool)
  embedded = self.embedding(start_tokens)
  return start_tokens, done, self.rnn.get_initial_state(embedded)[0]


@Decoder.add_method
def tokens_to_text(self, tokens):
  words = self.id_to_word(tokens)
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
  result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
  return result



@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature = 0.0):
  logits, state = self(
    context, next_token,
    state = state,
    return_state=True) 
  
  if temperature == 0.0:
    next_token = tf.argmax(logits, axis=-1)
  else:
    logits = logits[:, -1, :]/temperature
    next_token = tf.random.categorical(logits, num_samples=1)

  # If a sequence produces an `end_token`, set it `done`
  done = done | (next_token == self.end_token)
  # Once a sequence is done it only produces 0-padding.
  next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
  
  return next_token, done, state




# Setup the loop variables.
next_token, done, state = decoder.get_initial_state(ex_context)
tokens = []

for n in range(10):
  # Run one step.
  next_token, done, state = decoder.get_next_token(
      ex_context, next_token, done, state, temperature=1.0)
  # Add the token to the output.
  tokens.append(next_token)

# Stack all the tokens together.
tokens = tf.concat(tokens, axis=-1) # (batch, t)

# Convert the tokens back to a a string
result = decoder.tokens_to_text(tokens)
result[:3].numpy()



class Translator(tf.keras.Model):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, units,
               context_text_processor,
               target_text_processor):
    super().__init__()
    # Build the encoder and decoder
    encoder = Encoder(context_text_processor, units)
    decoder = Decoder(target_text_processor, units)

    self.encoder = encoder
    self.decoder = decoder

  def call(self, inputs):
    context, x = inputs
    context = self.encoder(context)
    logits = self.decoder(context, x)

    #TODO(b/250038731): remove this
    try:
      # Delete the keras mask, so keras doesn't scale the loss+accuracy. 
      del logits._keras_mask
    except AttributeError:
      pass

    return logits




model = Translator(UNITS, context_text_processor, target_text_processor)

logits = model((ex_context_tok, ex_tar_in))

print(f'Context tokens, shape: (batch, s, units) {ex_context_tok.shape}')
print(f'Target tokens, shape: (batch, t) {ex_tar_in.shape}')
print(f'logits, shape: (batch, t, target_vocabulary_size) {logits.shape}')



def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    
    return tf.reduce_sum(match)/tf.reduce_sum(mask)



model.compile(optimizer='adam',
              loss=masked_loss, 
              metrics=[masked_acc, masked_loss])


vocab_size = 1.0 * target_text_processor.vocabulary_size()


def check_unknowns(text, text_processor):
    # Normalize and preprocess the text to match the expected input format
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-zα-ωά-ώΑ-ΩΆ-Ώ.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)

    # Split the processed text into tokens
    tokens = tf.strings.split(text, ' ')

    # Flatten the tokens to ensure we have a 1D list of tokens
    tokens = tokens.flat_values if isinstance(tokens, tf.RaggedTensor) else tokens

    # Get the vocabulary and convert it to a Python set for fast lookup
    vocab_set = set(text_processor.get_vocabulary())

    # Convert each token to string and check if it's not in the vocabulary set
    unknown_tokens = [token.numpy().decode('utf-8') for token in tokens if token.numpy().decode('utf-8') not in vocab_set]

    return unknown_tokens





# Example usage
sample_texts = tf.constant(["Η τυπισα ειναι σαν γαμω την μανα σου Ειναι αυτουνου Κριμα που δεν τον καλεσα στο τετοιο Για 3; Θ το δεχτει Με τον γιαννη; Αυριο στις 10 θα παμε; Εχω φτασει Βαρεθηκα ορθοστασια Αν πηγες Και εσυ βαριοσουνα να πας Φαση σε καμια ωρα Σε ενα 10λεπτο ξεκινω Βεζω βενζυνα 20 μαξ Για βγες μπροστα Ειμαι φουλ κοντα"])
unknown_words = check_unknowns(sample_texts, target_text_processor)
print("Unknown words:", unknown_words)



{"expected_loss": tf.math.log(vocab_size).numpy(),
 "expected_acc": 1/vocab_size}


model.evaluate(val_ds, steps=20, return_dict=True)


history = model.fit(
    train_ds.repeat(), 
    epochs=100,
    steps_per_epoch = 100,
    validation_data=val_ds,
    validation_steps = 20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=4)])



#@title
@Translator.add_method
def translate(self,
              texts, *,
              max_length=50,
              temperature=0.0):
  # Process the input texts
  context = self.encoder.convert_input(texts)
  batch_size = tf.shape(texts)[0]

  # Setup the loop inputs
  tokens = []
  attention_weights = []
  next_token, done, state = self.decoder.get_initial_state(context)

  for _ in range(max_length):
    # Generate the next token
    next_token, done, state = self.decoder.get_next_token(
        context, next_token, done,  state, temperature)
        
    # Collect the generated tokens
    tokens.append(next_token)
    attention_weights.append(self.decoder.last_attention_weights)
    
    if tf.executing_eagerly() and tf.reduce_all(done):
      break

  # Stack the lists of tokens and attention weights.
  tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
  self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)

  result = self.decoder.tokens_to_text(tokens)
  return result




result = model.translate(['Που εισαι;']) # Are you still home
result[0].numpy().decode()



#@title
@Translator.add_method
def plot_attention(self, text, **kwargs):
  assert isinstance(text, str)
  output = self.translate([text], **kwargs)
  output = output[0].numpy().decode()

  attention = self.last_attention_weights[0]

  context = tf_lower_and_split_punct(text)
  context = context.numpy().decode().split()

  output = tf_lower_and_split_punct(output)
  output = output.numpy().decode().split()[1:]

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)

  ax.matshow(attention, cmap='viridis', vmin=0.0)

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + context, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + output, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  ax.set_xlabel('Input text')
  ax.set_ylabel('Output text')



model.plot_attention('¿Todavía está en casa?') # Are you still home


long_text = context_raw[-1]

import textwrap
print('Expected output:\n', '\n'.join(textwrap.wrap(target_raw[-1])))


model.plot_attention(long_text)


class Export(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
  def translate(self, inputs):
    return self.model.translate(inputs)

inputs = [
    'τι κανεις;', # "It's really cold here."
    'Πως εισαι;', # "This is my life."
    'Γεια' # "His room is a mess"
]


for t in inputs:
  print(model.translate([t])[0].numpy().decode())

print()
export = Export(model)

_ = export.translate(tf.constant(inputs))




tf.saved_model.save(export, 'translator',
                    signatures={'serving_default': export.translate})
