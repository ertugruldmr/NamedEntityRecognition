import gradio as gr
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from architrcture import CustomObjectScope, CustomNonPaddingTokenLoss

# File Paths
model_path = 'ner_model' 
mapping = {
    0: '[PAD]',
    1: 'O', 
    2: 'B-PER', 
    3: 'I-PER', 
    4: 'B-ORG', 
    5: 'I-ORG', 
    6: 'B-LOC', 
    7: 'I-LOC', 
    8: 'B-MISC', 
    9: 'I-MISC'
}

# defining the example texts
examples = [
    "CRICKET	-	LEICESTERSHIRE	TAKE	OVER	AT	TOP	AFTER	INNINGS	VICTORY",
    "Result	and	close	of	play	scores	in	English	county	championship	matches	on	Friday",
    "Adams	and	Platt	are	both	injured	and	will	miss	England	's	opening	World	Cup	qualifier	against	Moldova	on	Sunday",
    "Lying	three	points	behind	Alania	and	two	behind	Dynamo	Moscow	,	the	Volgograd	side	have	a	game	in	hand	over	the	leaders	and	two	over	the	Moscow	club"
]

# Load the model
with CustomObjectScope():
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss=CustomNonPaddingTokenLoss())

# lookup layer
vocabulary  = pickle.load(open("vocabulary", 'rb'))
lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)

def lowercase_and_convert_to_ids(tokens):
    tokens = tf.strings.lower(tokens)
    return lookup_layer(tokens)

def tokenize_and_convert_to_ids(text):
    tokens = text.split()
    return lowercase_and_convert_to_ids(tokens)

def formatting(text, prediction_labels):

    # tokenization
    prediction_tokens = text.split()

    # Generate HTML rendering with NER labels
    html = ""
    i = 0
    while i < len(prediction_tokens):
        
        # unpackaging
        token = prediction_tokens[i]
        label = prediction_labels[i]
        
        if label != "O":
            
            # setting the html
            html += "<span style='background-color: #ffff00; font-weight: bold;'>{} ({})</span>".format(token, label)
            j = i + 1
            while j < len(prediction_tokens) and prediction_labels[j] == label:
                html += " {}".format(prediction_tokens[j])
                j += 1
            html += " "
            
            i = j
        else:
            html += "{} ".format(token)
            i += 1
    return html

def predict(text):

  # encoding the text
  sample_input = tokenize_and_convert_to_ids(text)
  sample_input = tf.reshape(sample_input, shape=[1, -1])

  # prediction
  output = model.predict(sample_input)

  # decoding  the results
  prediction = np.argmax(output, axis=-1)[0]
  prediction = [mapping[i] for i in prediction]

  # html formatting for clear output 
  xml_reult = formatting(text, prediction)
  
  return xml_reult

# GUI Component
if_params = {
    "fn": predict ,
    "inputs": gr.inputs.Textbox(lines=5, label="Text"),
    "outputs": gr.outputs.HTML(label="NER Output"),
    "title": "Custom Named Entity Recognition Model",
    "description": "Enter a sentence to identify named entities in the text",
    "examples":examples
}

demo = gr.Interface(**if_params)

# Launching the demo
if __name__ == "__main__":
    demo.launch()
