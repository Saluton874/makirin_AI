#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#以下のサイトを参考にしました。
# https://qiita.com/pocket_kyoto/items/1e5d464b693a8b44eda5
# https://www.yoheim.net/blog.php?q=20160201

import sqlite3, os

path   = os.path.abspath(os.path.dirname(__file__))+'/'

conn = sqlite3.connect(path+"dataset/wnjpn-syn-database.1.0/wnjpn.db")

# 入力された単語(lemma)のWordIDを取得する
def getWordID(lemma):
    cur = conn.execute("select wordid from word where lemma='%s'" % lemma)
    for row in cur:
        wordid = row[0]
    return wordid


# WordIDから概念ID（synset）の一覧synsetsを作成する。
def getSynsetIDs(wordid):
    synsets = []
    cur = conn.execute("select synset from sense where wordid='%s'" % wordid)
    for row in cur:
        synsets.append(row[0])
    return synsets

## 概念ID（synset）から概念名を取得し表示する。使用していない。

#def getSynsetName(synset):
#    cur = conn.execute("select name from synset where synset='%s'" % synset)
#    for row in cur:
#        synset_name = row[0]
#    return synset_name

# 概念ID（synset）の意味（def）を取得する。
def getDefFromSynset(synset):
    cur = conn.execute("select def from synset_def where (synset='%s' and lang='jpn')" % synset)
    for row in cur:
        synset_def = row[0]
    return synset_def


# 概念ID(synset)を含む単語一覧(lemmasets）を取得する。つまり，類義語を取得する。

def getWordsFromSynset(synset,wordid):
    lemmasets = []
    cur1 = conn.execute("select wordid from sense where (synset='%s' and wordid!='%s' and lang='jpn')" % (synset,wordid))
    for row1 in cur1:
        tg_wordid = row1[0]
        cur2 = conn.execute("select lemma from word where wordid=%s" % tg_wordid)
        for row2 in cur2:
            lemmasets.append(row2[0])
    return lemmasets





def main():
    print("検索する単語を入力しエンターを押してください。\n終了する場合は×ボタンを押してください。")
    lemma = input()
    # 入力された単語が存在しない場合はスキップする。
    try:
        wordid = getWordID(lemma)
    except:
        print("\nこの単語は登録されていませんでした。\n")
        main()
    synsets = getSynsetIDs(wordid)
    counter1 = 1
    for synset in synsets:
        print("概念" + str(counter1) + "：" +getDefFromSynset(synset))
        counter1 += 1 
        lemmasets = getWordsFromSynset(synset,wordid)
        counter2 = 1
        for lemma in lemmasets:
            print("　類"+str(counter2)+"："+lemma)
            counter2 += 1
    print("\n検索結果は以上です。\n")
    main()


main()


