from bs4 import BeautifulSoup #this library is used for scrapping the dat from website
from urllib.request import urlopen,Request#this allow you to get the information form the internet

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = 'https://finviz.com/quote.ashx?t='

tickers = ['AMZN','TSLA','AAPL']


news_tables={}

for ticker in tickers:
    url=finviz_url+ ticker
    
    req= Request(url=url ,headers={'user-agent':'my_app'})
    response = urlopen(req)
    print(response)
    
    html = BeautifulSoup(response,'html')
    news_table = html.find(id="news-table")
    news_tables[ticker]=news_table
    
    

parsed_data= []

for ticker,news_table in news_tables.items():
    
    for row in news_table.findAll('tr'):
        
        title = row.a.get_text()
        date_data = row.td.text.split(" ")
        
        if len(date_data)==1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
            
        parsed_data.append([ticker,date,time,title])

print(parsed_data)

df = pd.DataFrame(parsed_data,columns=['ticker','date','time','title'])

print(df.head())

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound']=df['title'].apply(f)
print(df)


df['date']=pd.to_datetime(df.date).dt.date 

plt.figure(figsize=(10,8))

mean_df= df.groupby(['ticker','date']).mean()
mean_df=mean_df.unstack()
mean_df=mean_df.xs('compound',axis='columns').transpose()
mean_df.plot(kind='bar')
plt.show()
print(mean_df)

#print(vader.polarity_scores('Amazon is laying of its employees.is it due to ression or they are losing its market to etsy'))
# print(news_tables)

# amzn_data = news_tables['AMZN']#now we try to access all the news of amzaon 
# amzn_rows = amzn_data.findAll('tr')#finding all the data on amazon

# print(amzn_rows)

# for index,row in enumerate(amzn_rows):
#      title=row.a.text
#      timestamp = row.td.text
#      print(timestamp + " " + title)


#Now APPLYING SENTIMENTAL ANALYSIS

