# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import torch
import torch.optim as optim
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split

from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

config = Config(
    testing=True,
    bert_model_name="bert-base-uncased",
    max_lr=3e-5,
    epochs=4,
    use_fp16=True,
    bs=32,
    discriminative=False,
    max_seq_len=256,
)

from pytorch_pretrained_bert import BertTokenizer
bert_tok = BertTokenizer.from_pretrained(
    config.bert_model_name,
)

def _join_texts(texts:Collection[str], mark_fields:bool=False, sos_token:Optional[str]=BOS):
    """Borrowed from fast.ai source"""
    if not isinstance(texts, np.ndarray): texts = np.array(texts)
    if is1d(texts): texts = texts[:,None]
    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
    text_col = f'{FLD} {1} ' + df[0].astype(str) if mark_fields else df[0].astype(str)
    if sos_token is not None: text_col = f"{sos_token} " + text_col
    for i in range(1,len(df.columns)):
        #text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i]
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i].astype(str)
    return text_col.values

class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]


DATA_ROOT=Path('./data/')
df = pd.read_csv(DATA_ROOT / "dmoz_full_toplevel_imbalanced.tsv", sep='\t')

df = df.drop(columns=["externalpage_md5"])
df = df.drop(columns=["ages"])
df = df.drop(columns=["mediadate"])
df = df.drop(columns=["priority"])
df = df.drop(columns=["resource_md5"])
df = df.drop(columns=["language"])
df = df.drop(columns=["language_script"])
df = df.drop(columns=["language_iso6391"])
df = df.drop(columns=["language_confidence"])
df = df.drop(columns=["resource"])
df = df.drop(columns=["type"])
df = df.drop(columns=["topic"])
df = df.drop(columns=["topic_parent"])
try:
  df.drop(['Unnamed: 0'],axis=1,inplace=True)
except:
  pass

print(df.head())

# df_label= pd.read_csv(DATA_ROOT / 'label.csv')

print(df['topic_main'].value_counts())

print(df.head())

import imblearn
from imblearn.over_sampling import RandomOverSampler

y = df['topic_main']
X = df
# X = df.drop('topic_main', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

# before oversampling
print(pd.Series(y_train).value_counts()/len(y_train))

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X_train, y_train)

# print(X_ros.shape[0] - X.shape[0], 'new random picked points')
# print(pd.Series(y_ros).value_counts()/len(y_ros))

X_resampled = pd.DataFrame(X_ros, columns=X.columns)

print("\nX_resampled:\n")
print(X_resampled.head())
print(X_resampled.shape)

print("\ny_ros:\n")
print(y_ros.head())
print(y_ros.shape)

# after oversampling
print(pd.Series(y_ros).value_counts()/len(y_ros))

print(pd.Series(y_train).value_counts()/len(y_train))

train=pd.concat([X_resampled,pd.get_dummies(X_resampled['topic_main'])],axis=1)
train.sample(5)

del df

val=train

num_labels=y_ros.nunique()

if config.testing:
    train = train.head(1024)
    val = val.head(1024)
    # test = test.head(1024)

fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])

label_cols=y_ros.unique().tolist()
print(label_cols)

class BertTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)

class BertNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)

def get_bert_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    """
    Constructing preprocessors for BERT
    We remove sos/eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original BERT model.
    """
    return [BertTokenizeProcessor(tokenizer=tokenizer),NumericalizeProcessor(vocab=vocab)]

class BertDataBunch(TextDataBunch):
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_bert_processor)
        # use our custom processors while taking tokenizer and vocab as kwargs
        processor = get_bert_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)

databunch = BertDataBunch.from_df(".", train, val, None,
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_bert_vocab,
                  text_cols="topic_main",
                  label_cols=label_cols,
                  bs=config.bs,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )

from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=num_labels)

loss_func = nn.BCEWithLogitsLoss()

from fastai.callbacks import *
device_cuda = torch.device("cuda")
learner = Learner(
    databunch, bert_model,
    loss_func=loss_func
)
learner.model.cuda()
if config.use_fp16: learner = learner.to_fp16()
# learner.to_fp32()

learner.lr_find()

# learner.recorder.plot()

# learner.fit_one_cycle(config.epochs, max_lr=config.max_lr)
# learner.fit_one_cycle(config.epochs, max_lr=1e-02)
learner.fit_one_cycle(20, max_lr=config.max_lr)

train.catid.nunique()

learner.export(file=DATA_ROOT/'dmoz_model2.pkl')
learner.save(file=DATA_ROOT/'/dmoz_model2.pkl')

def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]

# learner.load(file=DATA_ROOT/'model1.pkl')
# test_preds = get_preds_as_nparray(DatasetType.Test)

# sample_submission = pd.read_csv(DATA_ROOT / "sample_submission.csv")
# if config.testing: sample_submission = sample_submission.head(test.shape[0])
# sample_submission[label_cols] = test_preds
# sample_submission.to_csv(DATA_ROOT /"predictions.csv", index=False)

# sample_submission=test
# for label in label_cols:
#   sample_submission[label]=0
# sample_submission[label_cols] = test_preds
# sample_submission.to_csv(DATA_ROOT /"predictions_1.csv", index=False)
# sample_submission.sample(10000).to_csv(DATA_ROOT /"predictions_sample.csv", index=False)

# sample_submission.head(5)

# output=sample_submission.head(1).drop(columns=['cat_id','category']).T
# output

# output[0]=output[0].astype(float)
# output.nlargest(5,[0])

# sample_submission.to_csv(DATA_ROOT /"predictions_test.csv", index=False)

# learner.export(file=DATA_ROOT + "/dmoz_model.pkl")
# learner.save(file=DATA_ROOT + '/dmoz_model.pkl')
learner.validate()

# text='Top Stories After Huge Win, Boris Johnson Promises Brexit By Jan 31, "No Ifs, No Buts" States Not Empowered To Block Citizenship Act, Say Government Sources Spent 66% Of Rs 3.38 Lakh Crore Budgeted Expenditure: Economic Advisor \'PM Should Apologise\': Rahul Gandhi Tweets Video Amid "Rape In India" Row "Won\'t Apologise," Says Rahul Gandhi Amid Row Over "Rape In India" Remark Watch: Sri Lanka Player\'s Hilarious Response To Pakistan Journalist More cricket Trending Watch: Steve "Flying" Smith Takes One Of The Best Catches You\'ll Ever See Reviews More Gadgets Reviews Samsung Galaxy A50, Galaxy A70, Galaxy S9 समत कई समसग समरटफन पर बपर डसकउट Samsung Galaxy M11 और Galaxy M31 अगल सल ह सकत ह लनच चर रयर कमर वल Samsung Galaxy A71 और Samsung Galaxy A51 लनच WhatsApp अगल सल स कई समरटफन पर नह करग कम PUBG Mobile in India May Get Privacy Destroying Features Will Nintendo\'s New Switch Consoles Be Better than the PS4 Pro?', 'Tamil Tamil परवततर म CAB क खलफ हसक परदरशन, गवहट म पलस क फयरग म 2 लग क मत गर BJP शसत रजय म CAB क वरध शर, अब पजब क सएम अमरदर सह न कह- बल असवधनक Aus Vs NZ: टम सउद न मर बललबज क गद, बच म भड गए वरनर, बल- \'उसक हथ म लग ह...\' दख Video UK Elections: एगजट पल म पएम बरस जनसन क कजरवटव परट क सपषट बहमत Bravo "Excited About Comeback" After Return To International Cricket Steven Gerrard Signs New Deal At Rangers Until 2024 Greenwood Stars As Man United Top Group, Arsenal Draw In Europa League November Trade Deficit Narrows To $12.', '12 Billion Food Bangladesh Asks India To Increase Guwahati Mission Security Amid Protests தமழ சனம நடகர சததரத அமசசர ஜயகமரகக பதலட..!', 'இநத வரம வளயகம எககசசகக தமழ படஙகள..!', "வபவ-இன டண' பட ரலஸ தத அறவபப..!", 'ஜய-அதலய ரவ ஜட சரம இரணடவத படம..!', 'டடடல வளயடட வறறமறன..!', 'Osteoporosis - Love your Bones Follow These Amazing Tips By Dr Kiran Lohia To Prevent Acne Breakouts Offbeat Baby Yoda To Disappointed Pakistani Fan: A Look At The Best Memes Of 2019 Biggest Parliament Majority For Boris Johnson\'s Party Since Thatcher Days South News No Top Court Order On Plea Of 2 Women For Protection To Enter Sabarimala Cities Nearly 7,000 Trees To Be Cut For Jewar Airport In Uttar Pradesh "We Have Been Cheated...": Teachers After Left Out Of Recruitment Process Campaigns 60,000 Blankets Needed: Help Save Lives, Donate A Blanket For The Homeless, Here\'s How Fighting Our Killer Air Pollution: Check The Air Quality Index Of Your City Chhattisgarh Becomes The Most Efficient State In Waste Management: Government A Startup In Uttarakhand Develops An Eco-Friendly Sanitary Pad That Lasts Five Times Longer Than Regular Pads'
# text='Efforts were being made on Friday to help passengers stranded at the airport, railway station and inter-state bus terminals in Guwahati, an official said.'
text = 'Eedama is a social enterprise that was formed as an independent training and consultancy agency to support companies, schools, universities and communities in their sustainability initiatives. Eedama was born from an environmental and green sensibility shared by scientists, engineers, teachers and parents with a common perception: talking about the environment is no longer enough, the time has come for true understanding and action.'
print('text:', text)

x = learner.predict(text)
y = pd.DataFrame(x[2], index=label_cols)
# x[2].shape
print(y.nlargest(5,[0]))

text='Efforts were being made on Friday to help passengers stranded at the airport, railway station and inter-state bus terminals in Guwahati, an official said.'
# text = 'Eedama is a social enterprise that was formed as an independent training and consultancy agency to support companies, schools, universities and communities in their sustainability initiatives. Eedama was born from an environmental and green sensibility shared by scientists, engineers, teachers and parents with a common perception: talking about the environment is no longer enough, the time has come for true understanding and action.'
print('text:', text)

x = learner.predict(text)
y = pd.DataFrame(x[2], index=label_cols)
# x[2].shape
print(y.nlargest(5,[0]))
