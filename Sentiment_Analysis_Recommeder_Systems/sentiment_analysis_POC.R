source("C:/Users/Sruthi/Desktop/My_StudyMaterial_PSG/RLabAssignments/sentiment_analysis_webscrapping.R")
library(tidytext)
library(dplyr)
library(tidyverse)
library(tidyr)
library(ggplot2)
library(glue)
files<-list.files("C:/Users/Sruthi/Desktop/My_StudyMaterial_PSG/RLabAssignments/project/")
#to fetch files from folder
GetSentiment<-function(file){
  fileName <- glue("C:/Users/Sruthi/Desktop/My_StudyMaterial_PSG/RLabAssignments/project/", file, sep = "")
  #to separate file name
  fileName <- trimws(fileName)
  #to remove white spaces in file name
  fileText <- glue(read_file(fileName))
  #to read large files
  fileText <- gsub("\\$", "", fileText)
  #to replace unwanted string
  tokens <- data_frame(text = fileText) %>% unnest_tokens(word, text)
  #to convert sentence to tokens(words)
  sentiment_bing<-tokens %>%
    inner_join(get_sentiments("bing"))
  #merge common tokens 
  sentiment_nrc<-tokens %>%
    inner_join(get_sentiments("nrc"))
  sentiment_nrc_sub <- tokens %>%
    inner_join(get_sentiments("nrc")) %>%
    filter(!sentiment %in% c("positive", "negative"))
  viewer <- sentiment_nrc %>%
    group_by(sentiment) %>%
    summarise(count = n()) %>%
    ungroup() %>%
    mutate(sentiment = reorder(sentiment, count)) %>%
    #fill = -count` to make the larger bars darker
    ggplot(aes(sentiment, count, fill = -count)) +
    geom_col() +
    theme_light() +
    labs(x = NULL, y = "Word Count") +
    scale_y_continuous(limits = c(0, 100)) + #Hard code the axis limit
    ggtitle("Story Sentiment") +
    coord_flip()
  plot(viewer)
  
}
for(i in files){
  sentiments <- rbind(sentiments, GetSentiment(i))
  summary(sentiments)
}