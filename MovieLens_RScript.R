##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#
#genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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

# We can have a look at dataset by following command.
glimpse(edx)

# Here we try to get more insight on the dataset. Following figures shows the distributions of the ratings.
ggplot(edx, aes(x=rating)) + geom_histogram() +
  geom_histogram(color="black", fill="white", binwidth = .5)

# Following two lines show that, there is 10677 movies and 69878 users in the dataset.
edx$movieId %>% unique() %>% length()
edx$userId %>% unique() %>% length()


# We can create a matrix which every row represents a user and every column represents a movie by following piece of code

if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
library(Matrix)

ratings_matrix <- sparseMatrix(i = edx$userId, j = edx$movieId , x = edx$rating)
dim(ratings_matrix)

# Each entry in this matrix is the rating given by a user to a movie. The matrix is very sparse. We can show this by visualizing the first 200 rows and columns.
userId <- 1:200
movieId <- 1:200
image(ratings_matrix[1:200,1:200])


# The easiest way is to take an average from all given ratings and assume that all the missing ratings are equal to this average value

avg_value <- mean(edx$rating)
avg_value

# we can calculate the rsme for this simple model as follows
actual <- validation$rating
predicted <- rep(avg_value,nrow(validation))
Metrics::rmse(actual,predicted)


# Here we use *recosystem* library. We first need to tune the model.  
# Following codes shows a suggestion. 
# The tuner goes through different combinations of the parameters and finds the best combination.

if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(recosystem)
train_reco <- with(edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_reco <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating))
reco <- Reco()

para_reco <- reco$tune(train_reco, opts = list(dim = c(20, 30),
                                               costp_l1 = 0,
                                               costq_l1 = 0,
                                               costp_l2 = c(0.01, 0.1),
                                               costq_l2 = c(0.01, 0.1),
                                               lrate = c(0.01, 0.1)))

# Presenting the results
print(para_reco)

reco$train(train_reco, opts = c(para_reco$min, nthread = 4, niter = 50))
predicted <- reco$predict(test_reco, out_memory())
Metrics::rmse(actual,predicted)

