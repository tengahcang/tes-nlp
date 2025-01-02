data='data_tweet.csv'
search_keyword='Kurikulum Merdeka'
limit=1000

npx tweet-harvest@latest -o "$data" -s "$search_keyword" -- $limit --token "<auth_token>"
