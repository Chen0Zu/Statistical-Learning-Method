clear;clc;close all;
dbstop if error
rng(1);
%% Generate data from two Gaussion distribution
N = 100;
mu1 = [1,1];
mu2 = [2,2];
sigma = 0.2 * eye(2);
X = [mvnrnd(mu1,sigma,N/2);mvnrnd(mu2,sigma,N/2)];
label = [ones(N/2,1); -ones(N/2,1)];
X_test = mvnrnd(mu1,sigma,1);

%
figure;
hold on;
plot(X(label==1,1),X(label==1,2),'ro');
plot(X(label==-1,1),X(label==-1,2),'bo');
plot(X_test(1,1),X_test(1,2), 'ko', 'markersize', 10);
hold off;
box on;
legend('Class 1', 'Class 2', 'Test data');

%% KNN
[D, I] = pdist2(X, X_test, 'euclidean', 'Smallest', 1);
pred_label = label(I);

%% KD Tree
