#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# http://ai-coordinator.jp/slackbot
# https://swdrsker.hatenablog.com/entry/2017/02/23/193137
# https://qiita.com/To_Murakami/items/cc225e7c9cd9c0ab641e
# https://kamo.hatenablog.jp/entry/2020/04/05/173810

import os
from gensim.models import KeyedVectors

path  = os.path.abspath(os.path.dirname(__file__))+'/'
kvs = KeyedVectors.load(path+'dataset/w2v_model/wiki.kv')
# wiki.kv.vectors.npy も必要

if __name__ == '__main__':
	while True:
		print(kvs.most_similar([input()], [], 1)[0][0])