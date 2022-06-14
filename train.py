#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import mod.as2s as as2s

path  = os.path.abspath(os.path.dirname(__file__))+'/'
model = as2s.model
N     = len(as2s.data)

# 学習
if __name__ == '__main__':
	st = datetime.datetime.now()
	for epoch in range(as2s.epoch_num):
		# ミニバッチ学習
		perm = np.random.permutation(N) # ランダムな整数列リストを取得
		total_loss = 0
	
		for i in range(0, N, as2s.batch_size):
			enc_words = as2s.data_converter.train_queries[perm[i:i+as2s.batch_size]]
			dec_words = as2s.data_converter.train_responses[perm[i:i+as2s.batch_size]]
			model.reset()
			loss = model(enc_words=enc_words, dec_words=dec_words, train=True)
			loss.backward()
			loss.unchain_backward()
			total_loss += loss.data
			as2s.opt.update()
	
		if (epoch+1)%10 == 0:
			ed = datetime.datetime.now()
			print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
			st = datetime.datetime.now()
	as2s.serializers.save_npz(path+'mod/dataset/as2s.npz', model)
