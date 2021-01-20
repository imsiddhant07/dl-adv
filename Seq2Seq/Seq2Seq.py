'''

-> SEQ2SEQ model implementation
for
-> MACHINE TRANSLATION
on the 
-> MULTI30K DATASET (German-To-English)
-> PyTorch

Part of : github.com/imsiddhant07/dl-adv

'''

##Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
#For pre-processsing :
from torchtext.data import Field, BucketIterator
#For logging metrics :
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
#For tokenizer :
import spacy

##Setting up for Pre-Processing
#Tokenizer
#Standard code for Germany-de || English-en
spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

#Funcation -> returns tokenized o/p
def tokenizer_ger(text):
	return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
	return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True,
	init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_eng, lower=True,
	init_token='<sos>', eos_token='<eos>')


##Splitting data into -> Train, Test, and Validation
train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
	fields=(german, english))

#Building vocabulary for language
#Parameters passed can be adjusted too, as HyperParameters
#max_size -> intented size of our vocabulary (can be a HyperParameter)
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


##Writing the model
#Encoder
class Encoder(nn.Module):
	pass

class Decoder(nn.Module):
	pass

class Seq2Seq(nn.Module):
	pass