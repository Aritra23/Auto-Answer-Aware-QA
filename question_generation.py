#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/patil-suraj/question_generation/blob/master/question_generation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


get_ipython().system('pip install -U transformers==3.0.0')


# In[ ]:


get_ipython().system('python -m nltk.downloader punkt')


# In[ ]:


get_ipython().system('git clone https://github.com/patil-suraj/question_generation.git')


# In[ ]:





# In[ ]:


text = "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum \
and first released in 1991, Python's design philosophy emphasizes code \
readability with its notable use of significant whitespace."

text2 = "Gravity (from Latin gravitas, meaning 'weight'), or gravitation, is a natural phenomenon by which all \
things with mass or energy—including planets, stars, galaxies, and even light—are brought toward (or gravitate toward) \
one another. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. \
The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing \
and forming stars and caused the stars to group together into galaxies, so gravity is responsible for many of \
the large-scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly \
weaker as objects get further away"

text3 = "42 is the answer to life, universe and everything."

text4 = "Forrest Gump is a 1994 American comedy-drama film directed by Robert Zemeckis and written by Eric Roth. \
It is based on the 1986 novel of the same name by Winston Groom and stars Tom Hanks, Robin Wright, Gary Sinise, \
Mykelti Williamson and Sally Field. The story depicts several decades in the life of Forrest Gump (Hanks), \
a slow-witted but kind-hearted man from Alabama who witnesses and unwittingly influences several defining \
historical events in the 20th century United States. The film differs substantially from the novel."


# ## Single task QA

# In[1]:


get_ipython().run_line_magic('cd', 'question_generation')


# In[2]:


from pipelines import pipeline


# In[ ]:


nlp = pipeline("question-generation")


# In[9]:


nlp(text3)


# If you want to use the t5-base model, then pass the path through model parameter

# In[ ]:


nlp = pipeline("question-generation", model="valhalla/t5-base-qg-hl")


# In[11]:


nlp(text3)


# In[12]:


nlp(text4)


# In[13]:


nlp(text2)


# In[ ]:





# ## Multitask QA-QG

# ### small-model

# In[5]:


nlp = pipeline("multitask-qa-qg")


# #### QG

# In[8]:


nlp(text)


# In[11]:


nlp(text2)


# In[10]:


nlp(text4)


# In[ ]:





# #### QA

# In[12]:


nlp({
  "question": "Who created Python ?",
  "context": text
})


# In[13]:


nlp({
    "question": "Who wrote Forrest Gump ?",
     "context": text4
})


# In[ ]:





# ### base-model

# In[14]:


nlp = pipeline("multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl")


# In[ ]:





# #### QG

# In[15]:


nlp(text)


# In[16]:


nlp(text2)


# In[17]:


nlp(text4)


# In[ ]:





# #### QA

# In[18]:


nlp({
  "question": "Who created Python ?",
  "context": text
})


# In[19]:


nlp({
    "question": "Who wrote Forrest Gump ?",
     "context": text4
})


# In[ ]:





# ## End-to-End QG

# ### small model

# In[4]:


nlp = pipeline("e2e-qg")


# In[5]:


nlp(text)


# In[6]:


nlp(text2)


# In[7]:


nlp(text4)


# In[ ]:





# ### base-model

# In[8]:


nlp = pipeline("e2e-qg", model="valhalla/t5-base-e2e-qg")


# In[9]:


nlp(text)


# In[10]:


nlp(text2)


# In[11]:


nlp(text4)


# In[ ]:




