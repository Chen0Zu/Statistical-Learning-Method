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

tree = ID3(X, label);
end

%%
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

%%
function tree = ID3(X, label, parent)

global tree_node;
global node_number;

if nargin == 2
    
    parent = 0;
    class = unique(label);
    node_number = 1;
    [n,d] = size(X);
    X(:,d+1) = (1:n)';
    node.fea_idx = 0;
    node.fea_set = 1:d;
    node.parent = 0;
    node.child_idx = 0;
    node.data_idx = 0;
    node.type = 'node';
    node.label = 0;
    
    cell_node(1:d) = node;
    tree_node = cell_node;
    clear cell_node;
    
    
else
    [n,d] = size(X(:,end-1));
    node_number = node_number + 1;
end

current = node_number;
tree_node(current).parent = parent;
tree_node(current).data_idx = X(:,end);

if all(label == label(1))
    tree_node(current).type = 'leaf';
    tree_node(current).label = label(1);
    tree_node(current).data_idx = X(:,end);
    tree = current;
    return;
end

[inf_gain, ~] = information_gain(X(:,1:end-1),label);
[~,idx] = max(inf_gain);
dim_value = unique(X(:,idx));
child = zeros(length(dim_value),1);

for i = 1:length(dim_value)
    split = X(:,idx) == dim_value(i);
    child(i) = ID3(X(split,:), label(split), current);
end

tree_node(current).child = child;

if nargin == 2
    tree = tree_node;
else
    tree = current;
end
end