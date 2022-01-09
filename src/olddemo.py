import sys
import fire
import json
import os
import numpy as np
import tensorflow as tf
import model, sample, encoder

import generate_unconditional_samples

import interactive_conditional_samples

class GPT2: 
  # bir əvvəlki əsasında bəzi mətn yaratmaq üçün mənbə kodu hasil
  def __init__(
      self,
      models_dir='models',
      model_name='1558M',
      seed=None,
      nsamples=1,
      batch_size=1,
      length=None,
      temperature=1,
      top_k=0,
      raw_text="",
  ):
      """
      İnteraktiv start modelləri
      :model_name=1558M : Simli, hansı model istifadə etmək
      :seed=None : Təsadüfi ədəd generatorları üçün tam başlanğıc dəyəri, oynatma üçün başlanğıc dəyərini düzəldin
       netice
      :nsamples=1 : Qaytarılan nümunələrin sayı cəmi
      :batch_size=1 : Paketlərin sayı (yalnız sürət / yaddaşa təsir edir). Psemply divedi bölmək lazımdır.
      :length=None : Yaradılan mətndə markerlərin sayı olmadıqda (default olaraq),
       modelin hiperparametrləri ilə müəyyən edilir
      :temperature=1 : Üzən point dəyəri, qəza nəzarət
       Boltsman paylanması. Aşağı temperatur təsadüfi sonluğu az miqdarda gətirib çıxarır. 
       Temperatur sıfıra yaxınlaşdıqca, model determinated olacaq və
       təkrarlanan. Yüksək temperatur daha çox təsadüfi sonluqlara gətirib çıxarır.
      :top_k=0 : Tamədədli məna, müxtəlifliyi idarə edir. 1 deməkdir
       hər bir addım üçün (mö ' cüzə) yalnız 1 sözü nəzərə alınır ki, bu da deterministic sonluqlara gətirib çıxarır,
       halbuki 40 Hər bir addımda 40 söz nəzərə alınır. 0 (default) - bu
       xüsusi özelleştirme, məhdudiyyətlərin olmaması deməkdir. 40, bir qayda olaraq, ən yaxşı dəyəri.
      """
      if batch_size is None:
          batch_size = 1
      assert nsamples % batch_size == 0

      self.nsamples = nsamples
      self.batch_size = batch_size
      models_dir = os.path.expanduser(os.path.expandvars(models_dir))
      self.enc = encoder.get_encoder(model_name, models_dir = 'models')
      hparams = model.default_hparams()
      with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
          hparams.override_from_dict(json.load(f))

      if length is None:
          length = hparams.n_ctx // 2
      elif length > hparams.n_ctx:
          raise ValueError("Artıq pəncərə ölçüsü nümunələri əldə edə bilməz: %s" % hparams.n_ctx)

      self.sess = tf.Session(graph=tf.Graph())
      self.sess.__enter__()
      
      self.context = tf.placeholder(tf.int32, [batch_size, None])
      np.random.seed(seed)
      tf.set_random_seed(seed)
      self.output = sample.sample_sequence(
          hparams=hparams, length=length,
          context=self.context,
          batch_size=batch_size,
          temperature=temperature, top_k=top_k
      )

      saver = tf.train.Saver()
      self.ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
      saver.restore(self.sess, self.ckpt)

  def close(self):
    self.sess.close()
  
  def generate_conditional(self,raw_text):
      context_tokens = self.enc.encode(raw_text)
      generated = 0
      for _ in range(self.nsamples // self.batch_size):
          out = self.sess.run(self.output, feed_dict={
              self.context: [context_tokens for _ in range(self.batch_size)]
          })[:, len(context_tokens):]
          for i in range(self.batch_size):
              generated += 1
              text = self.enc.decode(out[i])
              return text
              #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
              #print(text)
      #print("=" * 80)
###  
	  
gpt2 = GPT2(model_name="1558M")
###
class Who:
  """Söhbət iştirakçılarını müəyyən edən sinif: Mən, o"""
  def __init__(self):
    self.prefixes = []

  def matches(self,phrase):
    for prefix in self.prefixes:
      if phrase.startswith(prefix):
        #print(f"{phrase} starts with {prefix}")
        return True
      
    #print(f"{phrase} does not start with {self.prefixes}")
    return False

  def get_random_prefix(self):
    return self.prefixes[0]
  
class Me(Who):
  def __init__(self):
    super().__init__()
    self.prefixes = ["I said: \""]
   
  
class You(Who):
  def __init__(self):
    super().__init__()
    self.prefixes = ["You said: \""]

class Conversation:
  
  def __init__(self, prior = None):
    if prior is None:
      prior="""
      You said: "Nice to meet you. What's your name?"
      I said: "My name is Miri."
      You said: "That's an interesting name. How old are you?"
      I said: "I'm 19 years old."
      You said: "Can you tell me something about yourself?"
      I said: "Ofcourse! I like playing video games and eating cake. "
      You said: "I like sweet stuff too. What are your plans for tomorrow?"
      """
    self.suggestion = None
    
    self.me = Me()
    self.you = You()
    self.parties  = [ self.me, self.you ]
    
    self.conversation = []
    
    lines = prior.split("\n")
    for line in lines:
      line = line.strip()
      if len(line)!=0:
        party = None
        for party in self.parties:
          if party.matches(line):
            break
        if party is None:
          raise Exception(f"Unknown party: {line}")
                
        self.conversation.append((party,line))
    self.get_suggestion()
    
  
  def get_prior(self):
    conv = ""
    for (party, line) in self.conversation:
      conv+=line+"\n"
    return conv
  
  def get_suggestion(self):
    who, last_line = self.conversation[-1]

    party_index = self.parties.index(who)
    next_party = self.parties[(party_index+1) % len(self.parties)]
      
    conv = self.get_prior()
    conv += next_party.get_random_prefix()
    answer = self.get_answer(next_party, conv)

    if not next_party.matches(answer):
      prefix = next_party.get_random_prefix()
      answer = prefix + answer
    
    self.suggestion = (next_party, answer)
  
  def next(self, party = None, answer = ""):
    """Continue the conversation
    :param party: None -> use the current party which is currently in turn
    :param answer: None -> use the suggestion, specify a text to override the 
           suggestion
    
    """
    suggested_party, suggested_answer = self.suggestion
    if party is None:
      party = suggested_party
    
    if answer == "":
      answer = suggested_answer
      
    if not party.matches(answer):
      prefix = party.get_random_prefix()
      answer = prefix + answer
    
    answer = answer.strip()
    if answer[-1] != "\"":
      # add the closing "
      answer += "\""
      
    self.conversation.append((party, answer))    
    self.get_suggestion()
    
  def retry(self):
    self.get_suggestion()
        
  def get_answer(self, party, conv):
    answer = gpt2.generate_conditional(raw_text=conv)
    lines = answer.split("\n")
    line = ""
    for line in lines:
      if line !="":
        break
      
    if line!="":
      return line
    
    return ""
      
  def show(self):
    conv = ""
    for (party, line) in self.conversation:
      conv+=line+"\n"
    print(conv)
    if self.suggestion is not None:
      party, answer  = self.suggestion
      print("--> "+answer)
	  
	  
c = Conversation()
c.show()
c.retry()
c.show()
c.next()
c.show()
c.retry()
c.next(c.you, "Pizza is not to good for your health though.")
c.show()
gpt2.close()

# This is for possible future development but way slow out of date etc.
