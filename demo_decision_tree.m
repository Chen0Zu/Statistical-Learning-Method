function demo_decision_tree
%%
% Chen Zu Janurary 11, 2018
%
% It is a demo code for demonstrating decision tree algorithm.
%
% Related paper:
%
% [1] Hang Li. Statistical Learning Method, 2012.
%
%% Load Data
clc;clear;close all;
fid = fopen('贷款申请数据.csv','r');
title = textscan(fid, '%s %s %s %s %s %s',1,'delimiter', ',');
str = {'青年','中年','老年';
    '否','是','0';
    '否','是','0';
    '一般','好','非常好';
    '否','是','0'};
i = 1;
while ~feof(fid)
    data{i,1} = textscan(fid, '%s %s %s %s %s %s',1,'delimiter', ',');
    for j = 1:5
        switch data{i}{1,j+1}{1,1}
            case str{j,1}
                X(i,j)= 1;
            case str{j,2}
                X(i,j) = 2;
            case str{j,3}
                X(i,j) = 3;
        end
        
    end
    i = i + 1;
end
fclose(fid);

label = X(:,end);
X = X(:,1:end-1);

[inf_gain, inf_gain_ratio] = information_gain(X,label);
end

function [inf_gain, inf_gain_ratio] = information_gain(X,label)
[n,d] = size(X);
c = unique(label);
n_label = zeros(length(c),1);

H = 0;
for i = 1:length(c)
    n_label(i) = sum(label == c(i));
    H = H - n_label(i)/n*log2(n_label(i)/n);
end

cond_H = zeros(d,1);
for i = 1:d
    k = unique(X(:,i));
    for j = 1:length(k)
        t = 0;
        Di = sum(X(:,i)==k(j));
        for z = 1:length(c)
            Dik = sum(X(:,i) == k(j) & label == c(z));
            if Dik ~= 0
            t = t + Dik/Di*log2(Dik/Di);
            else
                t = t + 0;
            end
        end
        cond_H(i) = cond_H(i) - Di/n * t;
    end
end

inf_gain = H - cond_H;
inf_gain_ratio = inf_gain ./ H;
end
