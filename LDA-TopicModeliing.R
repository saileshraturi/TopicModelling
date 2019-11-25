setwd("/Users/saileshraturi/Desktop/Capastone")
terror_data = read.csv("TerrorismDataIndia2.csv",stringsAsFactors = FALSE)
attach(terror_data)

library(plyr)
library(dplyr)
data <- terror_data %>% select(motive) 
 motive_data <- unique(data)
head(motive_data, 100)

#Data Pre-Processing

motive_data = as.data.frame(motive_data)
motive_data = motive_data[!apply(motive_data == "", 1, all),]
motive_data = as.data.frame(motive_data)
motive_data = dplyr::mutate(motive_data, ID = row_number())


motive_data = as.data.frame(motive_data)

motive_data$motive_data <- sub("RT.*:", "", motive_data$motive_data)
motive_data$motive_data <- sub("@.* ", "", motive_data$motive_data)


install.packages("tidytext")
library(tidytext)
#tibble::rowid_to_column(motive_datanew$motive_data, "ID")

text_cleaning_tokens <- motive_data %>% 
  tidytext::unnest_tokens(word,motive_data)


text_cleaning_tokens$word <- gsub('[[:digit:]]+', '', text_cleaning_tokens$word)
text_cleaning_tokens$word <- gsub('[[:punct:]]+', '', text_cleaning_tokens$word)
text_cleaning_tokens <- text_cleaning_tokens %>% filter(!(nchar(word) == 1))%>% 
  anti_join(stop_words)
tokens <- text_cleaning_tokens %>% filter(!(word==""))
tokens <- tokens %>% mutate(ind = row_number())
tokens <- tokens %>% group_by(ID) %>% mutate(ind = row_number()) %>%
  tidyr::spread(key = ind, value = word)
tokens [is.na(tokens)] <- ""
tokens <- tidyr::unite(tokens, text,-ID,sep =" " )
tokens$text <- trimws(tokens$text)


#Model Building
#install.packages("tmcn")
#library(tmcn)
install.packages("textmineR")
library(textmineR)
dtm <- CreateDtm(tokens$text, 
                 doc_names = tokens$ID, 
                 ngram_window = c(1,2))



#explore the basic frequency
tf <- TermDocFreq(dtm = dtm)
original_tf <- tf %>% select(term, term_freq,doc_freq)

rownames(original_tf) <- 1:nrow(original_tf)
# Eliminate words appearing less than 2 times or in more than half of the
# documents
vocabulary <- tf$term[ tf$term_freq > 1 & tf$doc_freq < nrow(dtm) / 2 ]



dtm = dtm

#vocabulary <- frequencies$ncol  [ frequencies$ncol  > 1 & frequencies$ncol  < nrow(frequencies) / 2 ]
k_list <- seq(1, 10, by = 1)
#k_list = 50
model_dir <- paste0("models_", digest::digest(vocabulary, algo = "sha1"))
if (!dir.exists(model_dir)) dir.create(model_dir)
model_list <- TmParallelApply(X = k_list, FUN = function(k){
  filename = file.path(model_dir, paste0(k, "_topics.rda"))
  
  if (!file.exists(filename)) {
    m <- FitLdaModel(dtm = dtm, k = k, iterations = 500)
    m$k <- k
    m$coherence <- CalcProbCoherence(phi = m$phi, dtm = dtm, M = 5)
    save(m, file = filename)
  } else {
    load(filename)
  }
  
  m
}, export=c("dtm", "model_dir"))

#topic assignment



library(topicmodels)
k = 10
lda <- LDA(tokens$text, control = list(alpha = 0.1), 10)


model <- model_list[10][[ 1 ]]
gammadf = as.data.frame(model$gamma)
names(gammadf) = c(1:50)
gammadf

#for each doc find the best ranked topic
toptopics = as.data.frame(cbind(documents = tokens$text, topic = apply(gammadf,1,function(x) names(gammadf)[which(x==max(x))])))

#model tuning
#choosing the best model
coherence_mat <- data.frame(k = sapply(model_list, function(x) nrow(x$phi)), 
                            coherence = sapply(model_list, function(x) mean(x$coherence)), 
                            stringsAsFactors = FALSE)
library(ggplot2)
ggplot(coherence_mat, aes(x = k, y = coherence)) +
  geom_point() +
  geom_line(group = 1)+
  ggtitle("Best Topic by Coherence Score") + theme_minimal() +
  scale_x_continuous(breaks = seq(1,50,1)) + ylab("Coherence")

#model <- model_list[which.max(coherence_mat$coherence)][[ 1 ]]

model <- model_list[50][[ 1 ]]

model$top_terms <- GetTopTerms(phi = model$phi, M = 5)
top50_topic <- as.data.frame(model$top_terms)


model$topic_linguistic_dist <- CalcHellingerDist(model$phi)
model$hclust <- hclust(as.dist(model$topic_linguistic_dist), "ward.D")

model$hclust$clustering <- cutree(model$hclust, k = 6)

model$hclust$labels <- paste(model$hclust$labels, model$labels[ , 1])
plot(model$hclust)
# make a summary table
model$summary <- data.frame(topic     = rownames(model$phi),
                            cluster   = model$hclust$clustering,
                            model$labels,
                            coherence = model$coherence,
                            num_docs  = model$num_docs,
                            top_terms = apply(model$top_terms, 2, function(x){
                              paste(x, collapse = ", ")
                            }),
                            top_terms_prime = apply(model$top_terms, 2, function(x){
                              paste(x, collapse = ", ")
                            }),
                            stringsAsFactors = FALSE)


