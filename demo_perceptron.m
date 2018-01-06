function demo_perceptron
dbstop if error
rng(2);
%% Generate data from two Gaussion distribution
N = 10;
mu1 = [1,1];
mu2 = [2,2];
sigma = 0.2 * eye(2);
X = [mvnrnd(mu1,sigma,N/2);mvnrnd(mu2,sigma,N/2)];
label = [ones(N/2,1); -ones(N/2,1)];

%
figure;
hold on;
plot(X(label==1,1), X(label==1,2), 'ro');
plot(X(label==-1,1), X(label==-1,2), 'bo');
hold off;
box on;
title('Data');
xlabel('x1');
ylabel('x2');


%% Perceptron
eta = 0.1;
w = randn(2,1);
b = 0;
iter = 200;

% train
for i = 1:iter
    y = label .* (X*w + b);
    idx = find(y <= 0);
    if ~isempty(idx)
        j = randsample(idx,1);
        w = w + eta * label(j) * X(j,:)';
        b = b + eta * label(j);
    else
        break;
    end
end

% test
pre_label = sign(X*w+b);
accuracy = mean(sum(pre_label == label));

% Plot
hold on;
t = -0.5:0.1:2.5;
plot(t,(-b-w(1)*t)/w(2));
hold off;
legend('Class 1', 'Class 2','Hyperplane');
end