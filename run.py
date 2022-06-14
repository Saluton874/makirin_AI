#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import mod.as2s as as2s

path  = os.path.abspath(os.path.dirname(__file__))+'/'
model = as2s.model

as2s.serializers.load_npz(path+'mod/dataset/as2s.npz', model)

def predict(model, query):
	enc_query = as2s.data_converter.sentence2ids(query, train=False)
	dec_response = model(enc_words=enc_query, train=False)
	response = as2s.data_converter.ids2words(dec_response)
	print(query, "=>", response)

if __name__ == '__main__':
	#ここに感情など
	#処理
	predict(model, input())