"""Source: seaborn-data repo (diamonds dataset, 53,940 ashol hirer record)"""
import urllib.request

urllib.request.urlretrieve("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv", "data/diamonds.csv")
print("downloaded -> data/diamonds.csv")
