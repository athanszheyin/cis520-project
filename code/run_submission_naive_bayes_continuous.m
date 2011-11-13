%% Example submission: Naive Bayes Continuous

%% Load the data
load ../data/data_no_bigrams.mat;

% Make the training data
X = make_sparse(train);
Y = double([train.rating]');

% Run training
Yk = bsxfun(@eq, Y, [1 2 4 5]);
nb = nb_train_pk([X]'>0, [Yk]);

%% Make the testing data and run testing
Xtest = make_sparse(test, size(X, 2));
Yhat = nb_test_pk(nb, Xtest'>0);

%% Make predictions on test set

% Convert from classes 1...4 back to the actual ratings of 1, 2, 4, 5
Yhat = sum(bsxfun(@times, Yhat, [1 2 4 5]), 2);
save('-ascii', 'submit.txt', 'Yhat');
