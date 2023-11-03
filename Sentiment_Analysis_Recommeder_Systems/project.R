#Installing necessary packages
install.packages('rvest') 
install.packages('dplyr')

#Loading the packages
library(rvest)
library(dplyr)

#Specifying the url for desired website to be scraped
url = 'https://www.amazon.in/OnePlus-Nord-Lite-128GB-Storage/dp/B09WQYFLRX/ref=lp_1389401031_1_1?th=1'

#Reading the HTML code from the website
webpage = read_html(url)
print(webpage)

#Extracting the name of the product
title_html = html_nodes(webpage, 'h1#title')
title = html_text(title_html)
head(title)

#Extracting ratings of the product
rate_html <- html_nodes(webpage, 'span#acrPopover')
rate <- html_text(rate_html)
head(rate)

#Extracting user reviews
reviews = webpage %>% html_nodes(".review-text-content span") %>% html_text()
reviews

#Creating a data frame for user reviews
df = data.frame(reviews, stringsAsFactors = FALSE)

#Deleting unwanted rows from the data frame
df = df[-c(6,9),]

#Displaying the data frame as a table
View(df)

#Converting the data frame into .csv and .txt file formats
write.csv(df, "reviews.csv")
write.table(df, "reviews.txt")


 