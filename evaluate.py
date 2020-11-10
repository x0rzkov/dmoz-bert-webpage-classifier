# -*- coding: utf-8 -*-

from pathlib import Path

from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callback import *

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

fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])

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

DATA_ROOT=Path('./data/')

learn_inf = load_learner(path = DATA_ROOT, file='dmoz_model2.pkl')
learn_inf.loss_func = nn.BCEWithLogitsLoss()
if config.use_fp16: learn_inf = learn_inf.to_fp16()

text = 'Eedama is a social enterprise that was formed as an independent training and consultancy agency to support companies, schools, universities and communities in their sustainability initiatives. Eedama was born from an environmental and green sensibility shared by scientists, engineers, teachers and parents with a common perception: talking about the environment is no longer enough, the time has come for true understanding and action.'
# text='Top Stories After Huge Win, Boris Johnson Promises Brexit By Jan 31, "No Ifs, No Buts" States Not Empowered To Block Citizenship Act, Say Government Sources Spent 66% Of Rs 3.38 Lakh Crore Budgeted Expenditure: Economic Advisor \'PM Should Apologise\': Rahul Gandhi Tweets Video Amid "Rape In India" Row "Won\'t Apologise," Says Rahul Gandhi Amid Row Over "Rape In India" Remark Watch: Sri Lanka Player\'s Hilarious Response To Pakistan Journalist More cricket Trending Watch: Steve "Flying" Smith Takes One Of The Best Catches You\'ll Ever See Reviews More Gadgets Reviews Samsung Galaxy A50, Galaxy A70, Galaxy S9 समत कई समसग समरटफन पर बपर डसकउट Samsung Galaxy M11 और Galaxy M31 अगल सल ह सकत ह लनच चर रयर कमर वल Samsung Galaxy A71 और Samsung Galaxy A51 लनच WhatsApp अगल सल स कई समरटफन पर नह करग कम PUBG Mobile in India May Get Privacy Destroying Features Will Nintendo\'s New Switch Consoles Be Better than the PS4 Pro?'
# text='Efforts were being made on Friday to help passengers stranded at the airport, railway station and inter-state bus terminals in Guwahati, an official said.'
# text = 'When outfishing you can find some of the most interesting looking species, Especially those with giant fins and long noses. Sometimes out a sea you have absolutely no internet connection and you want to know what species of fish you just caught. With this deep learning fish classifier you take a picture of the fish and it will give you a solid prediction on that species.'
text2 = '''
Joe Biden has again said he is confident of victory as he inches closer to beating Donald Trump after Tuesday's US presidential election.

The Democratic challenger now has 253 of the 270 Electoral College votes needed to clinch the White House under the state-by-state US voting system.

Mr Biden also leads vote counts in the battlegrounds of Georgia, Nevada, Pennsylvania and Arizona.

A Biden win would see Mr Trump leave office in January after four years.
'''

x = learn_inf.predict(text2)
y = pd.DataFrame(x[2], index=['Arts', 'Business', 'Computers', 'Games', 'Health', 'Home', 'News', 'Recreation', 'Reference', 'Science', 'Shopping', 'Society', 'Sports'])

print('text:', text2)
print(y.nlargest(5,[0]))



