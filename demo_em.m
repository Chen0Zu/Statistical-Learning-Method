clc;
clear;
%% Initialization
pi = 0.4;
p = 0.6;
q = 0.7;
y = [1 1 0 1 0 0 1 0 1 1];

%%
Iter = 100;
for i = 1:Iter
    mu = pi*(p.^y).*((1-p).^(1-y))./...
        (pi*(p.^y).*((1-p).^(1-y))+(1-pi)*(q.^y).*(1-q).^(1-y));
    pi = mean(mu);
    p = mu*y'/sum(mu);
    q = (1-mu)*y'/sum(1-mu);
end