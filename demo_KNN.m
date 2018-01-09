function demo_KNN
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

% Show the toy data
figure;
hold on;
plot(X(label==1,1),X(label==1,2),'ro');
plot(X(label==-1,1),X(label==-1,2),'bo');
plot(X_test(1,1),X_test(1,2), 'ko', 'markersize', 10);
hold off;
box on;
legend('Class 1', 'Class 2', 'Test data');

%% KNN
K = 5;
[D, I] = pdist2(X, X_test, 'euclidean', 'Smallest', 5);
pred_label = sign(sum(label(I)));

%% Create KD Tree
data = [2,3; 5,4; 9,6; 4,7; 8,1; 7,2];
tree = create_KDTree(data);
end

%% KD Tree
function tree = create_KDTree(X, parent_number, split_dim)

global tree_node;
global node_number;

if nargin == 1
    [n, d] = size(X);
    X(:,d+1) = 1:n';
    node_number = 1;
    split_dim = 0;
    parent_number = 0;
    
    % initialize the node
    node.type = 'node';
    node.left = 0;
    node.right = 0;
    node.vector = zeros(1,d); % store the sample
    node.parent = 0;
    node.index = 0;
    node.numpoints = 0;
    
    % initialize the tree
    cell_node(1:n) = node;
    tree_node = cell_node;
    clear cell_node;
else
    [n,d] = size(X(:,1:end-1));
    node_number = node_number + 1;
end

assigned_nn = node_number; % assigned node number for this particular iteration

% set assignments to current node
tree_node(assigned_nn).parent = parent_number;

% only one data point left
if n == 1
    tree_node(assigned_nn).type = 'leaf';
    tree_node(assigned_nn).left = [];
    tree_node(assigned_nn).right = [];
    tree_node(assigned_nn).vector = X(:,1:end-1);
    tree_node(assigned_nn).index = X(:,end);
    tree_node(assigned_nn).numpoints = 1;
else
    
    % if there are more than 1 data point left
    variance = var(X(:,1:end-1));
    [~, split_dim] = max(variance);
    [~,idx] = sort(X(:,split_dim));
    pos = idx(floor(n/2)+1); % the position of median value
    
    tree_node(assigned_nn).type = 'node';
    
    % for feature value less than the median
    i = find(X(:,split_dim) < X(pos,split_dim));
    tree_node(assigned_nn).left = create_KDTree(X(i,:), assigned_nn, split_dim);
    
    j = X(:,split_dim) > X(pos,split_dim);
    if sum(j)>0
        tree_node(assigned_nn).right = create_KDTree(X(j,:), assigned_nn, split_dim);
    else
        tree_node(assigned_nn).right = [];
    end
    %
    tree_node(assigned_nn).vector = X(pos,1:end-1);
    tree_node(assigned_nn).index = X(pos,end);
    tree_node(assigned_nn).numpoints = n;
end

if nargin == 1
    tree = tree_node;
    clear global tree_node;
else
    tree = assigned_nn;
end
end

