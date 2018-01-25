clc;
clear;
close all;
rng(1);
% method = 'separate';
method = 'linear';
% method = 'kernel';

switch method
    case 'separate'
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
        
        %% dual separate SVM
        H = (X*X') .* (y*y');
        f = -ones(n,1);
        Aeq = y';
        beq = 0;
        lb = zeros(n,1);
        alpha = quadprog(H,f,[],[],Aeq,beq,lb,[]);
        [~,idx_dual] = sort(alpha,'descend');
        w_dual = X' * (alpha.*y);
        b_dual = y(idx_dual(1)) - X(idx_dual(1),:)*w_dual;
        
        %%
        t = 0.5:0.1:2.5;
        k = y.*(X*w+b);
        [~,idx] = sort(k);
        hold on;
        plot(t,(-b-w(1)*t)/w(2));
        plot(X(idx(1:4),1),X(idx(1:4),2),'s','markersize',10);
        hold off;
        
    case 'linear'
        mu1 = [1,1];
        mu2 = [2,2];
        sigma = 0.15 * [1,1];
        n = 50;
        X1 = mvnrnd(mu1, sigma, n);
        X2 = mvnrnd(mu2, sigma, n);
        X = [X1;X2];
        y = [ones(n,1);-ones(n,1)];
        [n,d] = size(X);
        hold on;
        plot(X1(:,1), X1(:,2),'rx');
        plot(X2(:,1),X2(:,2),'ko');
        hold off;
        box on;
        
        %% linear SVM
        C = 100;
        H = diag([ones(d,1);0;zeros(n,1)]);
        f = C*[zeros(d+1,1);ones(n,1)];
        Aineq = -[y.*X, y, eye(n)];
        bineq = -ones(n,1);
        lb = [-inf*ones(d+1,1); zeros(n,1)];
        variable = quadprog(H,f,Aineq,bineq,[],[],lb,[]);
        w = variable(1:d);
        b = variable(d+1);
        xi = variable(d+2:end);
        
        %% dual linear SVM
        C = 100;
        H = (X*X') .* (y*y');
        f = -ones(n,1);
        Aeq = y';
        beq = 0;
        lb = zeros(n,1);
        ub = C*ones(n,1);
        alpha = quadprog(H,f,[],[],Aeq,beq,lb,ub);
        [~,idx_dual] = sort(alpha,'descend');
        w_dual = X' * (alpha.*y);
        b_dual = y(idx_dual(1)) - X(idx_dual(1),:)*w_dual;
        
        %%
        t = 0.5:0.1:2.5;
        k = y.*(X*w+b);
        %         [~,idx] = sort(k);
        idx = xi > 1e-4;
        k(idx) = inf;
        [~,idx_sv] = sort(k);
        hold on;
        plot(t,(-b-w(1)*t)/w(2));
        plot(X(idx_sv(1:3),1),X(idx_sv(1:3),2),'s','markersize',15);
        plot(X(idx,1),X(idx,2),'s','markersize',10);
        hold off;
end
