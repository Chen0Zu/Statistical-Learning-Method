%%
% Chen Zu Janurary 9, 2018
%
% It is a demo code for demonstrating naivebayes algorithm.
%
% Related paper:
%
% [1] Hang Li. Statistical Learning Method, 2012.
%
%%
clc; clear; close all;
X = [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3;
    4 5 5 4 4 4 5 5 6 6 6 5 5 6 6]';
Y = [-1 -1 1 1 -1 -1 -1 1 1 1 1 1 1 1 -1]';
fea_set = [1 2 3; 4 5 6];
%%
[n,d] = size(X);
d1 = 3;
d2 = 3;
label = [1,-1];
c = length(label);
P_class1 = zeros(d1,c);
P_class2 = zeros(d1,c);
P = zeros(c,1);

% train
for i = 1:c
    P(i) = mean(Y == label(i));
    for j = 1:d1
        P_class1(j,i) = sum(X(Y == label(i),1) == fea_set(1,j)) / sum(Y == label(i));
        P_class2(j,i) = sum(X(Y == label(i),2) == fea_set(2,j)) / sum(Y == label(i));
    end
end

% test
x = [2,4]';
predp1 = P_class1(fea_set(1,:) == x(1),1)*P_class2(fea_set(2,:) == x(2),1)*P(1);
predp2 = P_class1(fea_set(1,:) == x(1),2)*P_class2(fea_set(2,:) == x(2),2)*P(2);

