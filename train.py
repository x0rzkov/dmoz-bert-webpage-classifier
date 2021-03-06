# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import coloredlogs, logging

import json
import string
import time
import os

import tqdm
import random_name

import torch
import torch.optim as optim
import torch.onnx

import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE 

from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification

from sklearn.model_selection import train_test_split

from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from fastai.metrics import *
# from fastai.metrics import error_rate, accuracy

import wandb
from wandb.fastai import WandbCallback

coloredlogs.install()

END = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[94m"
CYAN = "\033[96m"
DARKCYAN = "\033[36m"
GREEN = "\033[92m"
PURPLE = "\033[95m"
RED = "\033[91m"
UNDERLINE = "\033[4m"
YELLOW = "\033[93m"

SEED = 42
MONITOR = "accuracy"

LEARNING_RATE = 3e-3
LEARNING_RATE = slice(LEARNING_RATE)

FIND_LEARNING_RATE_BUDGET = 1
FIND_LEARNING_RATE_BUDGET = max(FIND_LEARNING_RATE_BUDGET, 1)

BERT_MODEL_NAME = "bert-large-uncased", # bert-base-uncased
SAVE_MODEL_PATH = "./models"
BASE_MODEL = BERT_MODEL_NAME

# Dataset split parameters
DATASET_SPLIT_TRAIN = 0.8
DATASET_SPLIT_VALID = 0.2
DATASET_SPLIT_TEST = 0.0

# Train parameters
TRAIN_EPOCHS = 1
TRAIN_METRIC = "error_rate"
TRAIN_BATCH_SIZE = 64

# WanDB variables
USE_WANDB = True
WANDB_PROJECT = "dmoz-classifier"
WANDB_API_KEY = "36928f7b58810b2b42194a7aba61b31745385b20"
# WANDB_USERNAME
# WANDB_NAME
# WANDB_NOTES
# WANDB_BASE_URL
# WANDB_MODE
# WANDB_TAGS
# WANDB_DIR
# WANDB_RESUME
# WANDB_RUN_ID
# WANDB_IGNORE_GLOBS
# WANDB_ERROR_REPORTING
# WANDB_SHOW_RUN
# WANDB_DOCKER
# WANDB_DISABLE_CODE
# WANDB_ANONYMOUS
# WANDB_CONFIG_PATHS
# WANDB_CONFIG_DIR
# WANDB_NOTEBOOK_NAME
# WANDB_HOST
# WANDB_SILENT

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

global wandb, wandb_run

if USE_WANDB:
    wandb.init(project=WANDB_PROJECT)
    wandb.config.allow_val_change=True
    wandb_run = wandb.init(job_type='train')
    wandb.config.epochs = TRAIN_EPOCHS
    wandb.config.batch_size = TRAIN_BATCH_SIZE
    wandb.config.entity = WANDB_ENTITY
    wandb.config.save_code = True
    wandb.config.name = date.today()

if round(DATASET_SPLIT_TRAIN + DATASET_SPLIT_VALID + DATASET_SPLIT_TEST, 5) != 1:
    sys.exit(
    """SPLIT RATIOS are not valid and should be equal to 1
    DATASET_SPLIT_TRAIN = %f
    DATASET_SPLIT_VALID = %f
    DATASET_SPLIT_TEST = %f
    """
        % (DATASET_SPLIT_TRAIN, DATASET_SPLIT_VALID, DATASET_SPLIT_TEST)
)

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

config = Config(
    testing=False,
    bert_model_name=BERT_MODEL_NAME,
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

def log(level, message):
    message = "[%s] %s" % (level.upper(), message)
    getattr(logging, level)(message)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class ExportModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, monitor:str='valid_loss', mode:str='auto', every:str='improvement', name:str='bestmodel', min_accuracy:float=0):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every,self.name = every,name
        self.best=min_accuracy
        if self.every not in ['improvement', 'epoch']:
            log("warning", f'ExportModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe export the model."
        if self.every=="epoch":
            log("info", f'ExportModel at epoch={epoch} because {self.every} found at {self.name}_{epoch}')
            self.learn.export(f'{self.name}_{epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if isinstance(current, Tensor): current = current.cpu()
            if current is not None and self.operator(current, self.best):
                log("info", f'Better model found at epoch {epoch} with {self.monitor} value: {current}. Exporting Learner {self.name}')
                self.best = current
                self.learn.export(f'{self.name}')

    def on_train_end(self, **kwargs):
        log("info", f'Export final epoch.')
        name = self.name.replace('.pkl', '')
        self.learn.export(f'{name}-final.pkl')

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))

def find_lr(learner, budget):
    log("info", f"{BOLD}Exploring to find a Learning Rate{END} with {BOLD}Budget = {budget}{END}")
    learner.fit_one_cycle(budget, max_lr=LEARNING_RATE)
    learner.unfreeze()

    try:
        log("info", f"Looking for a good {BOLD}Learning Rate{END}")
        learner.lr_find(stop_div=False, num_it=200)
        #learner.lr_find()
        fig = learner.recorder.plot(suggestion=True, return_fig=True)
        min_grad_lr = learner.recorder.min_grad_lr
        min_grad_lr = min_grad_lr
        log("info", f"{BOLD}Found min_grad_lr {min_grad_lr}{END}")
    except:
        new_budget = budget + 2
        log("info", f"{BOLD}Unable to find a Learning Rate{END} we will keep exploring with {BOLD}Budget = {new_budget}{END}")
        learner, min_grad_lr, budget = find_lr(learner, new_budget)

    learner.unfreeze()

    return learner, min_grad_lr, budget

def export_model(learner, filename):
    learner.model.eval();
    torch.save(learner.model, f'{filename}-default.pt')
    torch.save(learner.model.state_dict(), f'{filename}-state_dict.pt')

    torch.save(
        {
            "model_state_dict": learner.model.state_dict(),
        },
        f'{filename}-details.pt',
    )

    # for iOS app, we predict only 1 image at a time, we don't use batch
    # creating a dummy random input for graph input layer
    # with the following format (batch_size, nb_dimension (RGB), height, width)
    dummy_input = torch.randn(1, 3, TRAIN_IMG_INPUT_SIZE, TRAIN_IMG_INPUT_SIZE, requires_grad=False).cuda()

    log("info", f"Export {BOLD}ONNX model{END} to {BOLD}{filename}.onnx{END}")
    torch_out = torch.onnx._export(
                    learner.model,
                    dummy_input, 
                    f'{filename}.onnx',
                    verbose=True,
                    export_params=True
                    )

    learner.export(f'{filename}-export.pkl')
    learner.save(f'{filename}-save')

    return {'torch_default': f'{filename}-default.pt', 'onnx': f'{filename}.onnx', 'fastai_export': f'{filename}-export.pkl', 'fastai_save': f'{filename}-save.pth', 'torch_details': f'{filename}-details.pt', 'torch_state_dict': f'{filename}-state_dict.pt' }

def get_class_count(df, label_column, path_column):
    grp = df.groupby([label_column])[path_column].nunique()
    return {key: grp[key] for key in list(grp.keys())}
    
def get_class_proportions(df, label_column, path_column):
    class_counts = get_class_count(df, label_column, path_column)
    return {val[0]: round(val[1]/df.shape[0], 4) for val in class_counts.items()}

def split_df(df, validation_size, label_column, path_column):
    
    proportions = get_class_count(df, label_column, path_column)
    
    if DATASET_MAX_CLASSES_SIZE < 0:
      max_class_count = df[label_column].value_counts().max()
    else:
      max_class_count = DATASET_MAX_CLASSES_SIZE
      
    
    log("info", f"Analyzing classes ditribution for dataset")
    log("info", f"{BOLD}{DARKCYAN}max_class_count = {max_class_count}{END}")
    log("info", df[label_column].value_counts())
    
    log("info", f"{BOLD}Rebalancing dataset using oversampling strategy{END}")
    oversampled_df = pd.concat([y.sample(max_class_count, replace=True) for _, y in df.groupby(label_column)])
    log("info", oversampled_df[label_column].value_counts())
    
    train, validation = train_test_split(oversampled_df, test_size=validation_size, stratify=oversampled_df[label_column], random_state=SEED)
    
    train_class_proportions = get_class_proportions(train, label_column, path_column)
    validation_class_proportions = get_class_proportions(validation, label_column, path_column)
    
    log("info", f"{BOLD}Rebalanced Train data class proportions{END} {train_class_proportions}")
    log("info", f"{BOLD}Rebalanced Validation data class proportions{END} {validation_class_proportions}")
    
    train['is_valid'] = False
    validation['is_valid'] = True
    
    splitted_df = pd.concat([train, validation])
    log("info", f"{BOLD}Resulting auto-splitted dataset with stratification and rebalancing{END} {splitted_df}")
    return splitted_df

DATA_ROOT=Path('./data/')
DATASET_TSV_FILENAME="dmoz_full_toplevel_imbalanced.tsv"
DATASET_TSV_PATH=DATA_ROOT / DATASET_TSV_FILENAME

log("info", f"{BOLD}Loading dataset from TSV {DATASET_TSV_PATH}{END}")

df = pd.read_csv(DATASET_TSV_PATH, sep='\t')
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

# dataset_df.rename(
#     columns={
#         DATASET_CSV_COLUMN_PATH: "name",
#         DATASET_CSV_COLUMN_LABEL: "label",
#     },
#     inplace=True,
# )

print(df['topic_main'].value_counts())
print(df.head())

dataset_df = split_df(dataset_df, validation_size=DATASET_SPLIT_VALID, label_column="topic_name", path_column="description")
dataset_df.to_csv(DATASET_CSV_PATH.replace('.csv', '') + '-splitted.csv', index = False)

y = df['topic_main']
X = df
# X = df.drop('topic_main', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("\nX_train:\n")
print("\nX_train.head():\n")
print(X_train.head())
print("\nX_train.shape:\n")
print(X_train.shape)

print("\nX_test:\n")
print("\nX_test.head():\n")
print(X_test.head())
print("\nX_test.shape:\n")
print(X_test.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

# before oversampling
print("\nbefore oversampling:\n")
print(pd.Series(y_train).value_counts()/len(y_train))

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X_train, y_train)

# print(X_ros.shape[0] - X.shape[0], 'new random picked points')
# print(pd.Series(y_ros).value_counts()/len(y_ros))

X_resampled = pd.DataFrame(X_ros, columns=X.columns)

print("\nX_resampled:\n")
print("\nX_resampled.head():\n")
print(X_resampled.head())
print("\nX_resampled.shape:\n")
print(X_resampled.shape)

print("\ny_ros:\n")
print("\ny_ros.head():\n")
print(y_ros.head())
print("\ny_ros.shape:\n")
print(y_ros.shape)

# after oversampling
print("\nafter oversampling:\n")
print(pd.Series(y_ros).value_counts()/len(y_ros))

train=pd.concat([X_resampled,pd.get_dummies(X_resampled['topic_main'])],axis=1)
train.sample(5)

test=pd.concat([X_test,pd.get_dummies(X_test['topic_main'])],axis=1)
test.sample(5)

del df

val=train

num_labels=y_ros.nunique()

if config.testing:
    train = train.head(10240)
    val = val.head(10240)
    test = test.head(10240)

fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])

label_cols=y_ros.sort_values().unique().tolist()
# label_cols=y_ros.unique().sort(order='y').tolist()
# label_cols=y_ros.unique().tolist()
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

start_accuracy = 0

databunch = BertDataBunch.from_df(".", train, val, None,
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_bert_vocab,
                  text_cols="topic_main",
                  label_cols=label_cols,
                  bs=config.bs,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )

bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=num_labels)

loss_func = nn.BCEWithLogitsLoss()

learner = Learner(
    databunch, bert_model,
    loss_func=loss_func
)

best_model_path = f"best-{BASE_MODEL}" 
log("info", f"{BOLD}Best model{END} will be saved here : {BOLD}{best_model_path}{END}")
learner.callbacks.append(SaveModelCallback(learn, every='improvement', monitor=MONITOR, name=best_model_path))

learner_export_path = os.path.join(SAVE_MODEL_PATH, f"best-{BASE_MODEL}.pkl")
log("info", f"{BOLD}Exporting Learner{END} at {BOLD}{learner_export_path}{END}")
learner.callbacks.append(ExportModelCallback(learner, every='improvement', monitor=MONITOR, name= learner_export_path, min_accuracy=start_accuracy))

if USE_WANDB:
    log("info", f"{BOLD}Setting WanDB{END} to {BOLD}ON{END}")
    learner.callbacks.append(WandbCallback(learner, monitor=MONITOR))

learner.model.cuda()

if config.use_fp16: learner = learner.to_fp16()
# learner.to_fp32()

learner.lr_find()

# learner.recorder.plot()

# learner.fit_one_cycle(config.epochs, max_lr=config.max_lr)
# learner.fit_one_cycle(config.epochs, max_lr=1e-02)
learner.fit_one_cycle(8, max_lr=config.max_lr) # , callbacks=[SaveModelCallback(learner, every='improvement', monitor='accuracy', name='best_classifier_final')])

train.catid.nunique()

learner.export(file=DATA_ROOT/'dmoz_model4.pkl')
learner.save(file=DATA_ROOT/'/dmoz_model4.pkl')

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
