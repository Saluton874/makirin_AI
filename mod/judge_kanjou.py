#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://www.tdi.co.jp/miso/python-natural-language-processing

import MeCab
import pandas as pd
import csv, os

path   = os.path.abspath(os.path.dirname(__file__))+'/'

def kanjou(inp):
	## 今回使用するMeCabの出力フォーマット
	m = MeCab.Tagger('-F,%f[6] , -U,%m , -E,')
	
	#print(pn_df['Emotion'][19])s
	df_data = (inp)
	
	td = []
	
		
	## ダブルクォーテーション区切りする
	temp = m.parse(df_data).split(',')
	
	## 先頭と末尾の不要なデータを削除
	del temp[0]
	del temp[-1]
	
	## データの追加
	td.append(temp)
	
	## 日本語感情辞書の読み込み
	 
	df = pd.read_csv(path+"dataset/D18-2018.7.24/D18-2018.7.24.csv")
	pn_df = pd.read_csv(path+"dataset/D18-2018.7.24/PositiveNegativeSymbol.csv")
	
	
	## 日本語感情辞書を形態素解析する
	## 作業者シートのEmotionと感情分類のSymbolを紐付けて、
	## Wordの形態素解析結果, Emotion, PosNegのデータを作る
	 
	pnWord_dic=[]
	 
	 
	## 形態素解析対象のループ
	 
	for i in range (0,len(df),1):
		temp = []
		tempEmotion = []
		
		## Wordの形態素解析結果をダブルクォーテーション区切りにする
		
		temp.append(m.parse(df['Word'][i]).split(','))
		
		
		## Wordの先頭と末尾の不要なデータを削除
	
		del temp[0][0]
		del temp[0][-1]
		
		## 作業者シートのEmotionが複数の感情を持つデータ用にループ
		
		for j in range(0, len(df['Emotion'][i]),1):
	 
			
			## 作業者シートのEmotionが感情シートのSymbolに無い場合のエラー回避
			
			if (len(pn_df[pn_df.Symbol==df['Emotion'][i][j]])!=0):
				
				
				## 感情分類シートのEmotionとPosNeg抽出
				
				tempEmotion.append(pn_df[pn_df.Symbol==df['Emotion'][i][j]].values[0][0])
		
		
		temp.append(tempEmotion)
		
		
		## Emotionと形態素解析したWordを変数に代入
		pnWord_dic.append(temp)
	 
	pnWord_dic = pd.DataFrame(pnWord_dic, columns=['Word','Emotion'])
	
	
	temp = ['Word','Emotion']
	results = []
	results.append(temp)
	
	for j in range(0,len(td),1):
	
		temp = []
	
		Exp = []
		emotion = []
		
		## 日本語感情辞書をループ
		for i in range(0,len(pnWord_dic),1):
	
	
			## 日本語感情辞書の内容がTweetに含まれる場合
			if (set(pnWord_dic['Word'][i]).issubset(td[j])):
	
				## 感情を表現している言葉とその感情の抽出
				Exp.append(pnWord_dic['Word'][i])
				emotion.extend(pnWord_dic['Emotion'][i])
	
		## データレコードの生成
		temp.append(td)
		temp.append(emotion)
		
		results.append(temp)

	tt = pd.DataFrame(results, columns=[ 'Word','Emotion' ])
	
	## 176番目～177番目のデータが見やすいので、そこに表示を限定しています。
	
	return tt['Emotion'][1]