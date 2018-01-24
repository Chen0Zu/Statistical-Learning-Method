clc;
clear;
close all;
rng(1);
mu1 = [1,1];
mu2 = [2,2];
sigma = 0.05 * [1,1];
n = 50;
X1 = mvnrnd(mu1, sigma, n);
X2 = mvnrnd(mu2, sigma, n);
X = [X1;X2];
y = [ones(n,1);-ones(n,1)];
hold on;
plot(X1(:,1), X1(:,2),'rx');
plot(X2(:,1),X2(:,2),'ko');
hold off;
box on;

%% linear separate SVM
[n,d] = size(X);
H = diag([ones(d,1);0]);
A = -y.*[X,ones(n,1)];
b = -ones(n,1);
w = quadprog(H,[],A,b);
b = w(end);
w = w(1:end-1);

%%
t = 0.5:0.1:2.5;
k = y.*(X*w+b);
[~,idx] = sort(k);
hold on;
plot(t,(-b-w(1)*t)/w(2));
plot(X(idx(1:4),1),X(idx(1:4),2),'s','markersize',10);
hold off;