################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(tidyverse)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#Define the function that will calculate the RMSE
RMSE <- function(true_ratings, predicted_ratings){ sqrt(mean((true_ratings - predicted_ratings)^2)) }

#########################################################################
#Exploratory data analysis and vizualisation for users and movies ratings
#########################################################################

#Visualize the variability of mean rating for users with more than 100 ratings
edx %>% group_by(userId) %>% 
  summarize(mean = mean(rating)) %>% filter(n()>=100) %>% 
  ggplot(aes(mean)) + geom_histogram(bins = 30, color = "black")

#Visualize the variability of mean rating for movies with more than 100 ratings
edx %>% group_by(movieId) %>% 
  summarize(mean = mean(rating)) %>% filter(n()>=100) %>% 
  ggplot(aes(mean)) + geom_histogram(bins = 30, color = "black")

############################
#User and Movie effect model
############################

#Defining the "true" rating of the edx dataset by doing the mean of all its ratings
mu <- mean(edx$rating)

#Defining the movie bias
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) 

#Defining the user bias
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>% group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))

#Predict the ratings of the validation sets using the biases
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)

#Calculating the final RMSE between the predicted ratings and the actual ratings of the validation set
#for the movie and user biases model and rounding it to four decimal points 
model_user_movie <- round(RMSE(predicted_ratings, validation$rating),4)

#Creating a table with the rmses of different models
rmse_table <- data_frame(method = "Movie and user effect", RMSE = model_user_movie) 
print.data.frame(rmse_table)

###############################################################################
#Exploratory data analysis of best and worst prediction from the previous model
###############################################################################

#Connecting movie ID and titles
movie_titles <- edx %>% 
  select(movieId, title) %>% 
  distinct()

#The five movies with the best predictions and the number of time they have been rated
edx %>% count(movieId) %>% 
  left_join(movie_avgs) %>% 
  left_join(movie_titles, by="movieId") %>% 
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:5) 

#The five movies with the worst predictions and the number of time they have been rated
edx %>% count(movieId) %>% 
  left_join(movie_avgs) %>% 
  left_join(movie_titles, by="movieId") %>% 
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:5)


######################
# Regularization model
######################

#Divide the training set for cross validation purposes
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,] 
test_set <- edx[test_index,]

#Make sure all movies and users are common to the training and test set
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")

#Cross Validation aiming to find the best value for lambda for regularization
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  #Defining the "true" rating of the train set by doing the mean of all its ratings
  mu <- mean(train_set$rating)
  
  #Defining the movie bias using regularization
  b_i <- train_set %>% group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  #Defining the user bias using regularization
  b_u <- train_set %>% left_join(b_i, by="movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #Predict the ratings of the test set using the biases previously calculated
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    mutate(pred = mu + b_i + b_u) %>% pull(pred)
  
  #Return the RMSE between the predicted and actual ratings of the test set for every lambda
  return(RMSE(predicted_ratings, test_set$rating))
})

#Recuperating the best lambda from cross validation
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)] 

#Defining the "true" rating of the edx dataset by doing the mean of all its ratings
mu <- mean(edx$rating)

#Defining the movie bias for edx dataset using regularization and the best lambda
b_i <- edx %>% group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda))

#Defining the user bias for edx dataset using regularization and the best lambda
b_u <- edx %>% left_join(b_i, by="movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

#Predict the ratings of the validation set using the model previously created
prediction <- validation %>% 
  left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% pull(pred)

#Calculating the final RMSE between the predicted ratings and the actual ratings of the validation set
#for the regularization model and rounding it to four decimal points 
model_regularization <- round(RMSE(prediction, validation$rating),4)

#Adding the regularization model RMSE to the table
rmse_table <- bind_rows(rmse_table, data_frame(method="Regularized model", RMSE = model_regularization))
print.data.frame(rmse_table)
