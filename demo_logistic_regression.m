clc;
clear;
load iris_dataset.mat
class1 = irisTargets(1,:) ~= 0;
class2 = irisTargets(2,:) ~= 0;
X = irisInputs(:,class1 | class2)';
label = [ones(50,1);zeros(50,1)];

%%
iter = 100;
J = zeros(iter,1);
rng(1);
w = rand(5,1);
X = [ones(100,1),X];
eta = 0.001;
y = label;

for i = 1:iter
    hx = 1 - 1./(1+exp(X*w));
    J(i) = sum(y.*(X*w) - log(1+exp(X*w)));
    gw = sum(bsxfun(@times,(y - hx),X))';
    w = w + eta * gw;
end

%%
plot(J);