clear;clc;close all;
dbstop if error
rng(2);
%% Generate data from two Gaussion distribution
N = 10;
mu1 = [1,1];
mu2 = [2,2];
sigma = 0.2 * eye(2);
X = [mvnrnd(mu1,sigma,N/2);mvnrnd(mu2,sigma,N/2)];
label = [ones(N/2,1); -ones(N/2,1)];
% method = 'dual';
method = 'original';

%========================== show toy data ============================
figure;
hold on;
for i = 1:N
    if label(i) == 1
        linetype = 'ro';
    else
        linetype = 'bo';
    end
    plot(X(i,1), X(i,2), linetype);
    text(X(i,1)+0.05, X(i,2), num2str(i));
end
hold off;
box on;
title('Data');
xlabel('x1');
ylabel('x2');

if ~strcmp(method, 'dual')
    %% Perceptron
    eta = 0.1;
    w = zeros(2,1);
    b = 0;
    iter = 200;
    
    %================================== train ============================
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
        
        %% the code of this section for observing the change of hyperplan
        %     figure;
        %     hold on;
        %     for k = 1:N
        %         if label(k) == 1
        %             linetype = 'ro';
        %         else
        %             linetype = 'bo';
        %         end
        %         plot(X(k,1), X(k,2), linetype);
        %         text(X(k,1)+0.05, X(k,2), num2str(k));
        %     end
        %     t = -0.5:0.1:2.5;
        %     plot(t,(-b-w(1)*t)/w(2));
        %     hold off;
        %     box on;
        %     title('Data');
        %     xlabel('x1');
        %     ylabel('x2');
        
    end
else
    %% Dual of Perceptron
    eta = 0.1;
    K = X*X';
    alpha = zeros(N,1);
    b = 0;
    iter = 200;
    
    %================================== train ============================
    for i = 1:iter
        y = label.*(K*(alpha.*label) + b);
        idx = find(y <= 0);
        if ~isempty(idx)
            j = randsample(idx,1);
            alpha(j) = alpha(j) + eta;
            b = b + eta * label(j);
        else
            break;
        end
    end
    w = X'*(alpha.*label);
end

%============================= test =================================

pre_label = sign(X*w+b);
accuracy = mean(sum(pre_label == label));

% Plot hyperplane
hold on;
t = -0.5:0.1:2.5;
plot(t,(-b-w(1)*t)/w(2));
hold off;
legend('Class 1', 'Class 2','Hyperplane');