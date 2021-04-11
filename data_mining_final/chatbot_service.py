#!/usr/bin/env python
# coding: utf-8




import pandas as pd
movies_df = pd.read_csv("../Data/movie_pro.csv")
movies_df = movies_df.sort_values("n_rating", ascending=False)
movies_df = movies_df[movies_df.n_rating >= 100]
movies_df["upper_title"] = movies_df.title.apply(str.upper)

def munge_title(title):
    try:
        l = title.rfind('(') + 1
        year = int(title[l:l+4])
        i = title.rfind(' (')
        if i != -1:
            title = title[:i]
        for suff_word in ['The', 'A', 'An']:
            suffix = ', {}'.format(suff_word)
            if title.endswith(suffix):
                title = suff_word + ' ' + title[:-len(suffix)]
        return title+' ('+str(year)+')'
    except:
        return title

#movies_df['title'] = movies_df['title'].apply(munge_title)


# In[3]:


title_transform = {}
for i in list(movies_df.title):
    title_transform[i.upper()] = i


# In[4]:


import numpy as np
from tensorflow import keras
import tensorflow as tf
import json

#匯入模型
model = keras.models.load_model("./0109_Embedding_user_movie_30.h5")
model.trainable = False
model.compile(
    tf.optimizers.Adam(0.005),
    loss='MSE',
    metrics=['MAE'],
)
#model.summary()
temp = model.predict([np.array([3]).reshape(-1,1), np.array([3]).reshape(-1,1)])
del temp


embedd_movies = model.get_layer(name = "Embedd_movies")

(w,) = embedd_movies.get_weights()
movie_embedding_size = w.shape[1]

from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

#匯入資料
import pandas as pd
movies = movies_df

def munge_title(title):
    try:
        l = title.rfind('(') + 1
        year = int(title[l:l+4])
        i = title.rfind(' (')
        if i != -1:
            title = title[:i]
        for suff_word in ['The', 'A', 'An']:
            suffix = ', {}'.format(suff_word)
            if title.endswith(suffix):
                title = suff_word + ' ' + title[:-len(suffix)]
        return title+' ('+str(year)+')'
    except:
        return title

movies['title'] = movies['title'].map(munge_title)

moviename = movies.title

#movies = movies[movies.n_rating>1000]

movie_id_list = np.array(movies.movieId)
movie_id_len = len(movie_id_list)
title_list = movies['title'].copy()
title_list_total = pd.DataFrame(data = title_list)
title_list_total['movieId'] = np.array(movies.movieId)

#get similar
kv = WordEmbeddingsKeyedVectors(movie_embedding_size)
kv.add(
    movies['title'].values,
    w[movies.movieId]
)


# In[5]:



import pandas as pd
import numpy as np
import datetime
import time
import configparser
import logging
from os import listdir
import os
from os.path import isfile, join

from flask import Flask, request
import telegram
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
import requests
from datetime import datetime
import configparser
import dateparser

pd.options.mode.chained_assignment = None  # default='warn'
ISOTIMEFORMAT = '%Y%m%d_%H%M%S'

path = os.path.abspath('.')


# uvicorn filename:app --port 8001 --workers 5 --proxy-headers

from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, HTTPException, Query, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import uvicorn
import time



app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

   


# In[6]:


from urllib.parse import unquote, quote 
@app.get("/api/gsj_title")
async def similar_title_json(title:str="", nums:int=5):
    title = unquote(title)
    movie_title = np.array(moviename[moviename.str.contains(title)])[0]
    movieid = list(movies[movies.title == movie_title].movieId)[0]
    result = kv.most_similar(movie_title)
    target_movie_title_list = []
    movie_similar_rate = []
    target_movie_id_list = []
    for i in range(nums):
        target = result[i]
        target_movie_title_list.append(target[0])
        movie_similar_rate.append(target[1])
    out = {
        "input_movieId": movieid,
        "input_movie_title" : str(movie_title),
        "similar_movie_title_list" : target_movie_title_list,
        "similar_movie_similar_rate" : movie_similar_rate
    }
    return (out)


# In[7]:


def chatbot_similar_title_json(title:str="", nums:int=5):
    title = unquote(title)
    #movie_title = np.array(moviename[moviename.str.contains(title)])[0]
    movie_title = title
    movieid = list(movies[movies.title == movie_title].movieId)[0]
    result = kv.most_similar(movie_title)
    target_movie_title_list = []
    movie_similar_rate = []
    target_movie_id_list = []
    for i in range(nums):
        target = result[i]
        target_movie_title_list.append(target[0])
        movie_similar_rate.append(target[1])
    out = {
        "input_movieId": movieid,
        "input_movie_title" : str(movie_title),
        "similar_movie_title_list" : target_movie_title_list,
        "similar_movie_similar_rate" : movie_similar_rate
    }
    return (out)


# In[ ]:





# In[ ]:





# In[8]:


import pandas as pd
#movies_df = pd.read_csv("../Data/0913_movies.csv")

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

embs = np.load("embs.npy")
df = pd.read_csv("embedding_t_sne.csv")
df.title = df.title.apply(munge_title)
df["upper_title"] = df.title.apply(str.upper)



# Some helper functions for plotting annotated t-SNE visualizations

# TODO: adjust_text not available in kernels
try:
    from adjustText import adjust_text
except ImportError:
    def adjust_text(*args, **kwargs):
        pass

def adjust_text(*args, **kwargs):
    pass

def plot_bg(bg_alpha=.01, figsize=(13, 9), emb_2d=None):
    """Create and return a plot of all our movie embeddings with very low opacity.
    (Intended to be used as a basis for further - more prominent - plotting of a 
    subset of movies. Having the overall shape of the map space in the background is
    useful for context.)
    """
    if emb_2d is None:
        emb_2d = embs
    fig, ax = plt.subplots(figsize=figsize)
    X = emb_2d[:, 0]
    Y = emb_2d[:, 1]
    ax.scatter(X, Y, alpha=bg_alpha)
    return ax

def annotate_sample(n, n_ratings_thresh=0):
    """Plot our embeddings with a random sample of n movies annotated.
    Only selects movies where the number of ratings is at least n_ratings_thresh.
    """
    sample = df[df.n_ratings >= n_ratings_thresh].sample(
        n, random_state=1)
    plot_with_annotations(sample.index)

def plot_by_title_pattern(pattern, **kwargs):
    """Plot all movies whose titles match the given regex pattern.
    """
    match = df[df["upper_title"].apply(lambda x: (pattern).upper() in x)]
    return plot_with_annotations(match.index, **kwargs)

def add_annotations(ax, label_indices, emb_2d=None, **kwargs):
    if emb_2d is None:
        emb_2d = embs
    X = emb_2d[label_indices, 0]
    Y = emb_2d[label_indices, 1]
    ax.scatter(X, Y, **kwargs)

def plot_with_annotations(label_indices, text=True, labels=None, alpha=1, **kwargs):
    ax = plot_bg(**kwargs)
    Xlabeled = embs[label_indices, 0]
    Ylabeled = embs[label_indices, 1]
    if labels is not None:
        for x, y, label in zip(Xlabeled, Ylabeled, labels):
            ax.scatter(x, y, alpha=alpha, label=label, marker='1',
                       s=90,
                      )
        fig.legend()
    else:
        ax.scatter(Xlabeled, Ylabeled, alpha=alpha, color='green')
    
    if text and False:
        # TODO: Add abbreviated title column
        titles = df.loc[label_indices, 'title'].values
        texts = []
        for label, x, y in zip(titles, Xlabeled, Ylabeled):
            t = ax.annotate(label, xy=(x, y))
            texts.append(t)
        adjust_text(texts, 
                    #expand_text=(1.01, 1.05),
                    arrowprops=dict(arrowstyle='->', color='red'),
                   )
    return ax

FS = (13, 9)
def plot_region(x0, x1, y0, y1, text=True, mainpoint = None):
    """Plot the region of the mapping space bounded by the given x and y limits.
    """
    fig, ax = plt.subplots(figsize=FS)
    pts = df[
        (df.x >= x0) & (df.x <= x1)
        & (df.y >= y0) & (df.y <= y1)
    ]
    

    ax.scatter(pts.x, pts.y, alpha=.6)
    
    
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if text:
        texts = []
        for label, x, y in zip(pts.title.values, pts.x.values, pts.y.values):
            t = ax.annotate(label, xy=(x, y))
            texts.append(t)
        adjust_text(texts, expand_text=(1.01, 1.05))
    return ax

def plot_region_around(title, margin=5, **kwargs):
    """Plot the region of the mapping space in the neighbourhood of the the movie with
    the given title. The margin parameter controls the size of the neighbourhood around
    the movie.
    """
    xmargin = ymargin = margin
    match = df[df.title == title]
    print("TITLE", title)
    assert len(match) == 1
    row = match.iloc[0]
    
    return plot_region(row.x-xmargin, row.x+xmargin, row.y-ymargin, row.y+ymargin, mainpoint = match, **kwargs)


# In[ ]:





# In[9]:


import uuid


# In[ ]:





# In[ ]:





# In[10]:


def filename_plot_by_title_pattern(title):
    plot_by_title_pattern(title, figsize=(15, 9), bg_alpha=.05, text=False);
    filename = f"./tmp_data/{str(uuid.uuid1())}.png"
    plt.savefig(filename)
    return filename

def filename_plot_region_around(title):
    plot_region_around(title)
    filename = f"./tmp_data/{str(uuid.uuid1())}.png"
    plt.savefig(filename)
    return filename


# In[ ]:





# In[11]:


def transform_list_list(inputs):
    out = []
    for i in inputs:
        out.append([i])
    return out


# In[ ]:





# In[ ]:





# In[12]:


transform_list_list(["Harry Potter", "Captain", "Resident", "The Lord of"])


# In[ ]:





# In[13]:





Token = "Telegram Chatbot Token"

bot = telegram.Bot(token=Token)

from fastapi import FastAPI
from typing import Any, Dict, AnyStr, List, Union
JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]


from pydantic import BaseModel
class learning_curve_json(BaseModel):
    Timestamp: str
    Organizer: str
    Volunteer_name: str
    ActivityYear: str
    ActivityName: str
    StartTime: str
    EndTime: str
    ActivityLocation: str
    ActivityFeedback: str
    ActivityCategory: str
    YourEmail: str
    OnChain: str

class verify_app_json_format(BaseModel):
    app_id: int
    message_json: dict



@app.post('/hook')
async def webhook_handler(request: Request, arbitrary_json: JSONStructure = None):
    #print(arbitrary_json)
    """Set route /hook with POST method will trigger this method."""
    if request.method == "POST":
        #print("JJJJSON", arbitrary_json)
        keys = arbitrary_json.keys()
        for key in keys:
            
            if type(key) == bytes:
                arbitrary_json[key.decode()] = arbitrary_json[key]
                del arbitrary_json[key]
        update = telegram.Update.de_json(arbitrary_json, bot)
        dispatcher.process_update(update)
    return 'ok'




def start_handler(bot, update):
    print(bot)
    print(update)
    """Send a message when the command /start is issued."""
    
    start_message = "嗨，我是 Eason 開發的找相似電影機器人！\n請點擊「📡 開始找相似電影」開始使用。\n\n也歡應追蹤作者本人臉書 https://www.facebook.com/EasonC13/"
    
    update.message.reply_text(start_message, reply_markup = ReplyKeyboardMarkup([['📡 開始找相似電影']], resize_keyboard=True))

def help_handler(bot, update):
    """Send a message when the command /help is issued."""
    
    start_message = "嗨，我是 Eason 開發的找相似電影機器人！\n請點擊「📡 開始找相似電影」開始使用。\n\n也歡應追蹤作者本人臉書 https://www.facebook.com/EasonC13/"
    
    update.message.reply_text(start_message, reply_markup = ReplyKeyboardMarkup([['📡 開始找相似電影']], resize_keyboard=True))
    
def reset_handler(bot, update):
    reset_user(update)
    
def reply_handler(bot, update):
    
    print(update)
    AAA.append(update)
    
    text = update.message.text
    if text == "📡 開始找相似電影":
        tutorial = "請輸入電影名稱來找尋電影\n我會告訴你有哪些電影跟你找的電影很像\n也歡迎使用範例來開始"
        example = ReplyKeyboardMarkup([['Harry Potter'], ['Captain'], ['Resident'], ['Ring']], resize_keyboard=True)
        update.message.reply_text(tutorial, reply_markup = example)
        return True
    else:
        return process(bot, update, text)
    
def process(bot, update, text):
    bot.send_chat_action(chat_id=update.message.chat.id, action=telegram.ChatAction.TYPING)
    match_movies = movies_df[movies_df.title == text]
    if len(match_movies):
        for i in match_movies.title:
            title = i
        result = chatbot_similar_title_json(title, nums=10)
        outText = f"您找的電影是 {title}\n依序的電影名稱與相似度條列如下：\n\n"
        for i in range(len(result["similar_movie_similar_rate"])):
            tmp_title = result["similar_movie_title_list"][i]
            rate = result["similar_movie_similar_rate"][i]
            rate = round(rate, 3)
            outText += f"{i+1}. {tmp_title} : {rate}\n"

        outText += "\n 上圖為此電影在經過 T-SNE 降維後周圍的電影"
        filename = filename_plot_region_around(title)
        bot.send_photo(update.message.chat_id, photo=open(filename, 'rb'))
        update.message.reply_text(outText)

        os.remove(filename)
    else:
        contain_movies = list(movies_df[movies_df["upper_title"].apply(lambda x: text.upper() in x)].title)
        if contain_movies:
            if len(contain_movies) > 223:
                contain_movies = contain_movies[:223]
            filename = filename_plot_by_title_pattern(text)
            bot.send_photo(update.message.chat_id, photo=open(filename, 'rb'))
            update.message.reply_text("上圖為出現此電影名稱的電影在經過 T-SNE 降維後的分佈情形\n\n找到符合的電影條列如按鍵所示\n請問你要找哪一部的呢？", reply_markup = ReplyKeyboardMarkup(transform_list_list(contain_movies), resize_keyboard= True))
            os.remove(filename)
        else:
            update.message.reply_text("找不到電影，請再次輸入。\n", reply_markup = ReplyKeyboardMarkup([['📡 開始找相似電影']]))


    
def error_handler(bot, update, error = ""):
    """Log Errors caused by Updates."""
    print(update)
    import traceback
    import sys
    exc_type, exc_value, exc_tb = sys.exc_info()
    result = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(result)
    update.haveSend = True
    update.message.reply_text(text = '對不起，系統錯誤\n歡迎回報，告訴開發者 @EasonC13')


dispatcher = Dispatcher(bot, None)
dispatcher.add_handler(MessageHandler(Filters.text, reply_handler))
dispatcher.add_handler(CommandHandler('start', start_handler))
dispatcher.add_handler(CommandHandler('help', help_handler))
dispatcher.add_handler(CommandHandler('reset', reset_handler))
dispatcher.add_error_handler(error_handler)
#dispatcher.add_handler(CallbackQueryHandler(teacher_callback_handler, pattern='/teacherdata,'))


#https://api.telegram.org/bot1101986164:AAHRD1hDy0ZVStaYn9vxt4kZxOyLfubbVVA/setWebhook?url=https://gcp-wp-0.tsraise.com/hook

import threading
def doThreading(func, args, waitingTime = 0):
    time.sleep(waitingTime)
    t = threading.Thread(target = func, args = args)
    t.start()

domain = "fio10.ntnu.best"
def hook(domain = domain, token = Token):
    print(token)
    print(f"https://api.telegram.org/bot{token}/setWebhook?url={domain}/hook")
    time.sleep(1)
    result = requests.get(f"https://api.telegram.org/bot{token}/setWebhook?url={domain}/hook")
    print(result.json())

AAA = []
if __name__ == "__main__":
    # Running server
    doThreading(hook, {})
    #doThreading(hook, ('fio8.ntnu.best', "1344940898:AAGvj1VPKqxyQvkrU8KYWExxNBLzhXD_Ol8"))
    
    import nest_asyncio
    nest_asyncio.apply()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=13530)
    #app.run(port = 13527)


# ###### 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




