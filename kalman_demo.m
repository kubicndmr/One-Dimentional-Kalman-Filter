%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kalman Filter
%
% The problem: Predict the state vector of an arbitrary system
% having noisy measurements. 
%
% Assumptions:
% Noise is normally distributed, with mean 0 and known standard deviation and 
% considered systems are linear. 
%
% Model
%
% x[k] = A[k]*x[k-1] + B[k]*u[k] + w[k]  
% y[k] = C*x[k] + v[k]
%
% x[k] is the state vector we are interested in
%
% y[k] is the noisy observation of the state vector
%
% u[k] is the control vector and B is the corresponding filter
%
% A and C are transition matrices
%
% w and v are zero-mean Gaussian noises with covariances Q_k and R_k
% respectively
%
% Estimation consist of two steps:
%
% [1] Prediction
%
% x_hat[k|k-1] = A*x_hat[k-1|k-1] + B[k]*u[k]
% P[k|k-1]= A*P[k-1|k-1]*A^T+Q[k]
%
% P is the covariance matrix of the error signal e = (x[k]-x_hat[k])
% Note: We don't need explicitly x[k] here to compute P, that is computed
% recursive as well
%
% [2] Update
%
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear variables;
clc;

fs = 64; % Sample Rate
K = fs*10; % Total duration, 10 sec
L = fs*2.5; % Window duration
k = linspace(0,10,K);

%% Generate signal u
u = zeros(1,K);
u(1.5*fs:3*fs) = 1.5;
u(5*fs:6.5*fs) = 1;
u(7*fs:7.75*fs) = 2;
u(9*fs:9.5*fs) = 1;
u=u';

%% Generate Matrix B

% Arbitrary Impulse Response Representing Unknown System
B = zeros(1,L);
B((.25:.25:2)*fs) = [0.85 0.6 0.9 -0.6 -0.25 0.2 -0.1 0.15];

% Create a Gaussian Kernel 
g = ones(1,floor(fs*0.15));
for i=1:5
    b = ones(1,floor(fs*0.15));
    g = conv(g,b);
end
g = g./max(g);

% Smoothen Impulse Response and Generate Matrix B
Lg = length(g);

E = zeros(L,L);
for i=1:L-Lg
    E(i:i+Lg-1,i)=g;
end

B = E*B';

Lb = length(B);

E = zeros(K,K);
for i=1:K-Lb
    E(i:i+Lb-1,i)=B;
end

B = E;

%% Generate A and C
A = 1;
C = 1;

%% Generate Gaussian Noises w and v
Q = 0.25;
R = 1;
w = normrnd(0,Q,K,1);
v = 0.1*normrnd(0,R,K,1);

%% Compute State and Observations Vector

x = ones(K,1);
x = A*x+B*u+w;
x = x./max(x);
y = C*x+v;

%% Visualize 
plot(k,x,'Linewidth',3);
hold on;
plot(k,y,'r','Linewidth',1.25)
grid on;
xlabel('Time (t)')
ylabel('Magnitude')
title('X');
legend('State Variable','Noisy Observation')

%% Initialize P and x_hat

x_hat = zeros(K+1,1);
P = ones(K+1,1);

%% Recursion

for k = 2:K
    %Prediction
    x_hat(k) = A*x_hat(k-1)*A+B(k,:)*u;
    P(k) = A*P(k-1)*A+Q;
    
    %Update
    KG = P(k)*C/(C*P(k)*C+R);
    x_hat(k) = x_hat(k) + KG*(y(k)-C*x_hat(k));
    P(k)=P(k)-KG*C*P(k);
    
end
x_hat = x_hat(2:end,:);
x_hat = x_hat./max(x_hat);

k = linspace(0,10,K);
error = x-x_hat;

figure;
plot(k,x_hat,'Linewidth',3);
hold on;
plot(k,error,'y','Marker','+');
grid on;
xlabel('Time (k)')
ylabel('Magnitude')
title('$\hat{x}$','Interpreter','latex')
legend('Estimated State Signal','Error With True Value')