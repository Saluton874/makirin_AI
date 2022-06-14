#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
口調変更
'''
import spacy, re, random
nlp = spacy.load('ja_ginza')

def changer(sentence):
	sentence=re.sub(r'しないと','しなきゃ',sentence)
	sentence=re.sub(r'(友達|友人)','お友達',sentence)
	sentence=re.sub(r'いかがでしょう','どうでしょう',sentence)
	sentence=re.sub(r'(?!ござ)います',random.choice(['う','うかも']),sentence)
	sentence=re.sub(r'ごめんなさい',random.choice(['ごめんね','ごめん……']),sentence)
	sentence=re.sub(r'ございます',random.choice(['ね','……','ございます']),sentence)
	sentence=re.sub(r'からです',random.choice(['もんね','もん','から']),sentence)
	sentence=re.sub(r'でください',random.choice(['でね','でくださいね','でください……']),sentence)
	sentence=re.sub(r'のです',random.choice(['んです','んだ']),sentence)
	sentence=re.sub(r'いかが',random.choice(['いかが','どう']),sentence)
	sentence=re.sub(r'かねます','られません',sentence)
	sentence=re.sub(r'ません',random.choice(['ません','ないです']),sentence)
	sentence=re.sub(r'かも.*','かも'+random.choice(['','です']),sentence)
	sentence=re.sub(r'ます',random.choice(['ちゃいます','ます']),sentence)
	sentence=re.sub(r'しまい',random.choice(['しちゃい','しまい']),sentence)
	sentence=re.sub(r'いです',random.choice(['いです','いね']),sentence)
	sentence=re.sub(r'です.*',random.choice(['です','かも','かなぁ']),sentence)
	sentence=re.sub(r'でしょう',random.choice(['でしょうか','かも','かもです','でしょう','かなぁ']),sentence)

	# 正規化
	sentence=re.sub(r'(花陽|はなよ|ハナヨ|かよ|カヨ)(ち(ゃ){0,1}ん|チ(ャ){0,1}ン){0,1}','ハナヨ',sentence)
	sentence=re.sub(r'(凛|りん|リン)(ちゃん|チャン){0,1}','リン',sentence)
	sentence=re.sub(r'凜(ちゃん|チャン){0,1}','渋谷凛',sentence)
	sentence=re.sub(r'((にこ|ニコ)(ッチ|っち|にー|ニー)|((世界の){0,1}矢澤|ヤザワ)|((宇宙){0,1}ナンバーワンアイドル))(ちゃん|チャン){0,1}','ニコ',sentence)
	sentence=re.sub(r'(真姫|まき|マキ)(ちゃん|チャン){0,1}','マキ',sentence)
	sentence=re.sub(r'(穂乃果|ほのか|ほの|ほの|はのけ|ハノケ)(ちゃん|チャン|ちゅん|ちぇん){0,1}','ホノカ',sentence)
	sentence=re.sub(r'(海未|うみ|ウミ|ウミミ|うみみ)(ちゃん|チャン){0,1}','ウミ',sentence)
	sentence=re.sub(r'(ことり|コトリ)(ちゃん|チャン){0,1}','コトリちゃん',sentence)
	sentence=re.sub(r'(希|のぞみ|ノゾミ|のん|ノン)(ちゃん|チャン|たん|タン){0,1}','ノゾミ',sentence)
	sentence=re.sub(r'(絵里|えり|エリ)(ちゃん|チャン|ち|チ){0,1}','エリ',sentence)
	sentence=re.sub(r'(米|コメ|こめ)(デブ|でぶ)(ちゃん|チャン){0,1}','【悪口】ハナヨ',sentence)
	sentence=re.sub(r'(くそ|クソ)(猫|ねこ|ネコ)(ちゃん|チャン){0,1}','【悪口】リン',sentence)
	sentence=re.sub(r'((ほの|ホノ)(かす|カス))(ちゃん|チャン){0,1}','【悪口】ホノカ',sentence)
	sentence=re.sub(r'((こと|コト)(かす|カス))(ちゃん|チャン){0,1}','【悪口】コトリ',sentence)
	sentence=re.sub(r'((ほの|ほむ|ホノ|ホム)(まん|マン))','穂むら饅頭',sentence)
	
	sentence=re.sub(r'(.*)(は|が){0,1}(馬鹿|ばか|バカ|塵|ごみ|ゴミ|屑|くず|クズ|阿呆|あほ|アホ|デブ|でぶ(死|し|シ|タヒ)(ね|ネ))','【悪口】\\1',sentence)
	

	doc = nlp(sentence)

	SETTINGS = {
		'名前':'花陽',
		'相手':'あなた',
		'趣味':'',
		'好きな食べ物':'',
		'好きな人':'',
		'友達':['マキちゃん','リンちゃん','ホノカちゃん','コトリちゃん','ウミちゃん','ノゾミちゃん','ニコちゃん','エリちゃん']
	}

	make = ''
	flag = False
	data = []
	target = []
	emo_flag = []
	for sent in doc.sents:
		for token in sent:
			ja_pos = token.tag_.split('-')
			# flag操作
			if len(ja_pos) > 1 and ja_pos[1] in '括弧開': flag = True
			if len(ja_pos) > 1 and ja_pos[1] in '括弧閉': flag = False
			if len(ja_pos) > 1 and ja_pos[1] in ['括弧開','括弧閉']: continue

			if flag == True:
				if len(ja_pos) > 1 and ja_pos[1] == '係助詞': flag = False
				if token.orth_ == '悪口': emo_flag.append('悪口')
				continue

			if len(ja_pos) > 2 and ja_pos[2] == '副詞可能':
				data.append(token.lemma_)
				flag = True
				continue

			if len(ja_pos) > 1 and ja_pos[1] == '固有名詞':
				target.append(token.orth_)
				make += token.orth_
				if sent[token.i+1].tag_.split('-')[0] != '接尾辞': make += 'ちゃん'
				continue

			if ja_pos[0] == '接尾辞':
				target.append(sent[token.i-1].lemma_)
				make += 'ちゃん'
				continue

			if token.lemma_ == '沢山':   make += 'いっぱい'
			elif token.lemma_ == '少し':   make += 'ちょっと'
			elif token.lemma_ == '迚も':   make += 'とっても'
			elif token.lemma_ == '大きな': make += 'おっきな'
			elif token.lemma_ == '大きい': make += 'おっきい'
			elif token.lemma_ == '私': make += SETTINGS['名前']
			else: make += token.orth_
			#print(token.i, token.orth_, token.lemma_, token.pos_,token._.pos_detail, token.dep_, token.head.i)
	#print(make,emo_flag,target,data)
	return make

if __name__ == '__main__':
	sentence=input()
	print(changer(sentence))