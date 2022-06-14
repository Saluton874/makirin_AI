#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# -----------------------------------
# Listen to my heart!!
# -----------------------------------

from mod.text_normalize import normalize as nor
from mod.judge_category import cat_and_point as cat
from mod.judge_kanjou   import kanjou
from mod.judge_negaposi import negaposi
from mod.judge_plutchik import plutchik 


class Medical:
	def __init__(self, inp):
		self.inp = nor(inp)
		self.nowcat = []
		self.ngps = 0
		self.plutchik = {}

	def category_check(self):
		cate = cat(self.inp)
		if cate[1] < 5:
			self.nowcat.append(['未分類',0])
		else:
			self.nowcat.append([cate[0].split('-')[0],cate[1]])
		if len(self.nowcat) > 3:
			self.nowcat = self.nowcat[0:3]
		return self.nowcat

	def kanjou_check(self):
		self.plutchik = plutchik(kanjou(self.inp))
		return self.plutchik

	def negaposi_check(self):
		self.ngps += negaposi(self.inp)
		return self.ngps
	
	def main(self):
		# カテゴリに近い返答を探す。
		# 現在の会話カテゴリ、同カテゴリが2回続いたら、Wikiからカテゴリっぽいことを独り言？再度カテゴリが変わるか、独り言するまでフラグ切れない。
		# 【感情】感情によって言い方を変える（プルチックがマイナスのものは読み込まない）
		# ネガポジはどうしよう一定数を越したら・・・モデル変更？5くらいを目安？
		# SPAMは暴言・エロなどのチェック。よくわかりません、聞き取れませんでしたなどを返す
		
		# あと質問チェッカー入れたい。think.pyから分離させよう
		# SPQMチェックと同じロジックで判定するのと、Mecabの形態解析の2つでどうにかする
		
		
		
		
		
		# この流れはmental.pyでやるか悩みどころ。
		# mental.pyは、あくまでも心の分析にすべきか、悩んでいる。

		#                                                目的語抽出
		# 会話の流れ.                       ┌───────────────────────────────────────┐
		# 小泉花陽ちゃんはかわいいよね　→ sim星空凛ちゃんも可愛いです → 誰「それ」？ → wiki要約 星空凛ちゃんはラブライブ!の人物です。
		# マキリンちゃんの自動生成文章に目的語がない場合、「それ」が使われている場合かつ相手に目的語がある場合。
		# は、も、はW2V.pyで最も似ているものを選出（W2V使用はランダム）
		# なるべくテニヲハを意識してもらう。
		print()


#print(Medical(input()).kanjou_check())
test = Medical(input())
print(test.kanjou_check())
