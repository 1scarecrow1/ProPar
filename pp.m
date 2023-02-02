%% Generating time varying parameters and observations
% This section specifies the time-varying process of generating parameters,
% and subsequently observations.

mu = [0.5 ; 0; 0 ; 0.5]; % mean of vec(A_t)
rho = 0.9; % autoregression coefficient
mu_eta = [0; 0; 0; 0]; %mean of shock in time-varying process
sigma_eta = [[0.001, 0, 0, 0]; [0, 0.001, 0, 0]; [0, 0, 0.001, 0]; [0, 0, 0, 0.001]]; % covariance matrix of eta_t

mu_eps = [0; 0]; %mean of shock in observation-generating process
sigma_eps = [0.05 0; 0 0.05]; %covariance matrix of epsilon_t

% Generate the initial value of vec(A_t)
vecA = mu;
A = reshape(vecA, [2,2]);

tvpvecA = vecA; %store time varying parameters in vec form
tvpA = A; %store time varying parameters in matrix form

tvpvecAcons = mu; %store time-varying parameters with linear constraint
tvpAcons = reshape(mu, [2,2]); %matrix form

y = [0;0]; %initial observation of y_t at t = 1
obs = y;

y_cons = [0;0];
obs_cons = y_cons; %store observations generated under constraints

% Generate a time series of vec(A_t)
T = 1000; % number of time steps

for t = 2:T
    eta = mvnrnd(mu_eta, sigma_eta); % generate a sample of eta_t
    eps = mvnrnd(mu_eps, sigma_eps); % generate a sample of epsilon_t
    vecA = mu + rho*(vecA - mu) + reshape(eta, [4,1]);
    A = reshape(vecA, [2,2]);
    y = A*y + reshape(eps, [2,1]);
    tvpvecA = [tvpvecA vecA]; 
    tvpA = [tvpA A];
    obs = [obs y];
    
    vecAcons = vecA ./ sum(vecA); %sum of elements in constrained vector = 1
    Acons = reshape(vecAcons, [2,2]);
    y_cons = Acons*y_cons + reshape(eps, [2,1]);
    tvpvecAcons = [tvpvecAcons vecAcons];
    tvpAcons = [tvpAcons Acons];
    obs_cons = [obs_cons y_cons];
end

a1 = tvpvecA(1,:);
time = 1:1:T;
plot(time, a1)
xlabel('Time')
ylabel('a1')
title('Parameter a1')

a3 = tvpvecA(2,:);
plot(time, a3)
xlabel('Time')
ylabel('a3')
title('Parameter a3')

a2 = tvpvecA(3,:);
plot(time, a2)
xlabel('Time')
ylabel('a2')
title('Parameter a2')

a4 = tvpvecA(4,:);
plot(time, a4)
xlabel('Time')
ylabel('a4')
title('Parameter a4')

y1 = obs(1,:);
plot(time, y1)
xlabel('Time')
ylabel('y1')
title('Series y1')

y2 = obs(2,:);
plot(time, y2)
xlabel('Time')
ylabel('y2')
title('Series y2')

%% ProPar update - non-batch processing


P = 10*eye(4); %penalty matrix 4x4 identity matrix
vecAupd = mu; %time t = 1
updates = vecAupd; %store updates

for t = 2:T
    x = obs(:,t-1);
    X = kron(eye(2), x.');
    vecApred = mu + rho*(vecAupd - mu);
    vecAupd = inv(X.'*inv(sigma_eps)*X + P) * (P.' * vecApred + X.' * inv(sigma_eps)*obs(:,t));
    updates = [updates vecAupd];
end

upd_a1 = updates(1,:);
plot(time, upd_a1, time, a1)
xlabel('Time')
ylabel('a1')
title('ProPar Updates of a1 vs True Parameter')
legend('ProPar updates', 'True Parameter a1')

upd_a2 = updates(2,:);
plot(time, upd_a2, time, a2)
xlabel('Time')
ylabel('a2')
title('ProPar Updates of a2 vs True Parameter')
legend('ProPar updates', 'True Parameter a2')

upd_a3 = updates(3,:);
plot(time, upd_a3, time, a3)
xlabel('Time')
ylabel('a3')
title('ProPar Updates of a3 vs True Parameter')
legend('ProPar updates', 'True Parameter a3')

upd_a4 = updates(4,:);
plot(time, upd_a4, time, a4)
xlabel('Time')
ylabel('a4')
title('ProPar Updates of a4 vs True Parameter')
legend('ProPar updates', 'True Parameter a4')

for t = 2:T
    mse_a1 = immse(tvpvecA(1,:), upd_a1);
    mse_a2 = immse(tvpvecA(2,:), upd_a2);
    mse_a3 = immse(tvpvecA(3,:), upd_a3);
    mse_a4 = immse(tvpvecA(4,:), upd_a4);
end

%% ProPar update - batch processing


batch_size = 50;
batch_vecAupd = mu;
batch_updates = batch_vecAupd;

for i = 2:T/batch_size
    start_batch = (i-1)*batch_size + 1;
    end_batch = i*batch_size;
    obs_batch = obs(:,start_batch:end_batch);
    X_batch = kron(eye(2), obs_batch(:,1:batch_size-1).');
    batch_vecApred = mu + rho*(batch_vecAupd - mu);
    batch_vecAupd = inv(X_batch.'*inv(sigma_eps)*X_batch + P)*(P.'*batch_vecApred + X_batch.'*inv(sigma_eps)*obs_batch(:,batch_size));
    batch_updates = [batch_updates batch_vecAupd]; 
end


%% Linear constraint observations

% mu_cons = [0.5 ; 0; 0 ; 0.5]; % mean of vec(A_t)
% rho_cons = 0.9; % autoregression coefficient
% mu_eta_cons = [0; 0; 0; 0]; %mean of shock in time-varying process
% sigma_eta_cons = [[0.001, 0, 0, 0]; [0, 0.001, 0, 0]; [0, 0, 0.001, 0]; [0, 0, 0, 0.001]]; % covariance matrix of eta_t
% 
% % Generate the initial value of vec(A_t)
% vecAcons = mu_cons;
% A_cons = reshape(vecAcons, [2,2]);
% 
% tvpvecA_cons = vecAcons; %store time varying parameters in vec form
% tvpAcons = A_cons; %store time varying parameters in matrix form
% 
% y_cons = [0;0]; %initial observation of y_t at t = 1
% obs_cons = y_cons;
% 
% for t = 2:T
%     eta_cons = mvnrnd(mu_eta_cons, sigma_eta_cons); % generate a sample of eta_t
%     eps = mvnrnd(mu_eps, sigma_eps); % generate a sample of epsilon_t
%     vecAcons = mu_cons + rho_cons*(vecAcons - mu_cons) + reshape(eta_cons, [4,1]);
%     vecAcons = vecAcons ./ sum(vecAcons);
%     A_cons = reshape(vecAcons, [2,2]);
%     y_cons = A_cons*y + reshape(eps, [2,1]);
%     tvpvecA_cons = [tvpvecA_cons vecAcons]; 
%     tvpAcons = [tvpAcons A_cons];
%     obs_cons = [obs_cons y_cons];
% end
% 
% X_init_const = (kron(eye(2), obs_cons(:,1).'));
% omega_init_const = X_init_const.'*inv(sigma_eps)*X_init_const + P;
% vecAupd_init_const = mu_cons + inv(omega_init_const)*ones(4,1)*(inv(ones(1,4)*inv(omega_init_const)*ones(4,1)))*(1 - ones(1,4)*mu_cons);
% const_updates = vecAupd_init_const;
% 
% for t = 2:T
%    X_const =(kron(eye(2), obs_cons(:,t).'));
%    omega_const = X_const.'*inv(sigma_eps)*X_const + P;
%    vecAupd_const = tvpvecA(:,t) + inv(omega_con)*ones(4,1)*inv(ones(1,4)*inv(omega_con)*ones(4,1))*(1 - ones(1,4)*tvpvecA(:,t));
%    const_updates = [const_updates vecAupd_const];
% end
% 

%% Update with linear constraints

X_init = (kron(eye(2), obs(:,1).'));
omega_init = X_init.'*inv(sigma_eps)*X_init + P;
vecAupd_const = mu + inv(omega_init)*ones(4,1)*(inv(ones(1,4)*inv(omega_init)*ones(4,1)))*(1 - ones(1,4)*mu);
cons_updates = vecAupd_const;

for t = 2:T
   X_cons =(kron(eye(2), obs(:,t).'));
   omega_cons = X_cons.'*inv(sigma_eps)*X_cons + P;
   vecAupd_cons = tvpvecA(:,t) + inv(omega_cons)*ones(4,1)*inv(ones(1,4)*inv(omega_cons)*ones(4,1))*(1 - ones(1,4)*tvpvecA(:,t));
   cons_updates = [cons_updates vecAupd_cons];
end

upd_cons_a1 = cons_updates(1,:);
plot(time, upd_cons_a1, time, upd_a1, time, a1)
xlabel('Time')
ylabel('Updated constrained a1')
title('ProPar Updates of a1')
legend('constrained updates', 'unconstrained updates', 'true parameter a1')


%% Constant A matrix
% Actual A matrix is constant, while the batch size is small/constant, while the 
% penalty matrix is of order t, such that the penalty matrix increases over time. Then our estimate 
% should over time convergence to the true matrix 

n = 50000;
n_periods = 1:1:n;
constantA = [0.5 0.1; 0.1 0.5]; %true A
y_constantA = [0;0]; %initial observation
constantAobs = y_constantA; %store observations
epsilon = mvnrnd(mu_eps, sigma_eps, n - 1); %generate normal errors

for t = 2:n
    y_constantA = constantA*y_constantA + reshape(epsilon(t-1,:), [2,1]); %generate observations
    constantAobs = [constantAobs y_constantA]; %store observations
end

vecAupd_constantA = [-0.7;0.8;-0.2;0.9]; %assume wrong initial parameters 
updates_constantA = vecAupd_constantA; %store updates

updates_constantA = nan(4,n);

for t = 2:n
    x_constantA = constantAobs(:,t-1);
    X_constantA = kron(eye(2), x_constantA.');
    newP = 30*eye(4)*t/100; %penalty matrix of order t
    vecAupd_constantA = inv(X_constantA.'*inv(sigma_eps)*X_constantA + newP) * (newP.' * vecAupd_constantA + X_constantA.' * inv(sigma_eps)*constantAobs(:,t));
    updates_constantA(:,t) = vecAupd_constantA;
end

figure 
subplot(2,2,1)
plot(n_periods, updates_constantA(1,:)) 
ylim([0 0.6]);
yline(constantA(1,1))
xlabel('Time')
ylabel('Updated a1 for constant A')
title('ProPar Updates of a1')
legend('updates', 'true a1')

subplot(2,2,2)
plot(n_periods, updates_constantA(2,:)) 
ylim([0 0.6]);
yline(constantA(1,2))
xlabel('Time')
ylabel('Updated a2 for constant A')
title('ProPar Updates of a2')
legend('updates', 'true a2')

subplot(2,2,3)
plot(n_periods, updates_constantA(3,:)) 
ylim([0 0.6]);
yline(constantA(2,1))
xlabel('Time')
ylabel('Updated a3 for constant A')
title('ProPar Updates of a3')
legend('updates', 'true a3')

subplot(2,2,4)
plot(n_periods, updates_constantA(4,:)) 
ylim([0 0.6]);
yline(constantA(2,2))
xlabel('Time')
ylabel('Updated a4 for constant A')
title('ProPar Updates of a4')
legend('updates', 'true a4')
%% Penalty long-run mse filtering
% Set penalty to be multiple of the identity. Then plot Long-run filtering MSE as a function of the penalty 
% parameter, which controls the whole penalty matrix. This MSE has a minimum for some penalty parameter that 
% makes the right trade-off between responsiveness and stability 

p_min = 0; % minimum penalty parameter value
p_max = 50; % maximum penalty parameter value
step = 1; % step size
p_range = p_min:step:p_max; % range of penalty parameter values

n_p = 1000;
mse_vecAupd = mu; %time t = 1
mse_vecupdates = mse_vecAupd; %store updates
mse_upd = mse_vecupdates;
% Loop over the penalty parameter values
for i = 1:length(p_range)
    penalty = p_range(i)*eye(4); % Set the penalty matrix (P) for each iteration 
    mse_vecAupd = mu;
    mse_vecupdates = mse_vecAupd;
    for t = 2:n_p
        x_ = obs(:,t-1);
        X_ = kron(eye(2), x_.');
        vecApred_ = mu + rho*(mse_vecAupd - mu);
        mse_vecAupd = inv(X_.'*inv(sigma_eps)*X_ + penalty) * (penalty.' * vecApred_ + X_.' * inv(sigma_eps)*obs(:,t));
        mse_vecupdates = [mse_vecupdates mse_vecAupd];
    end
    mse_upd = [mse_upd mse_vecupdates];
    longrunmse(i) = mse(mse_vecupdates(end),tvpvecA);
    
end

% Plot the Long-run filtering MSE as a function of the penalty parameter
plot(p_range, longrunmse);
xlabel('Penalty parameter');
ylabel('Long-run filtering MSE');

%% Deterministic A 
% DESCRIPTIVE TEXT

y_sinA = [0;0]; %initial observation
sinAobs = y_sinA; %store observations
epsilon = mvnrnd(mu_eps, sigma_eps, n - 1); %generate normal errors
sinAparams = nan(4,n);

for t = 2:n
    sinA = [sin(t) (pi/4)*sin(t) ;(pi/4)*sin(t) sin(3*pi/t)];
    y_sinA = sinA*y_sinA + reshape(epsilon(t-1,:), [2,1]); %generate observations
    sinAobs(:,t) = y_sinA; %store observations
    sinAparams(:,t) = reshape(sinA, [4,1]);
end

vecAupd_sinA = [-0.7;0.8;-0.2;0.9]; %assume wrong initial parameters 
updates_sinA = vecAupd_sinA; %store updates

updates_sinA = nan(4,n);

for t = 2:n
    x_sinA = sinAobs(:,t-1);
    X_sinA = kron(eye(2), x_sinA.');
    new_P = 30*eye(4)*t/100; %penalty matrix of order t
    vecAupd_sinA = inv(X_sinA.'*inv(sigma_eps)*X_sinA + new_P) * (new_P.' * vecAupd_sinA + X_sinA.' * inv(sigma_eps)*sinAobs(:,t));
    updates_sinA(:,t) = vecAupd_sinA;
end

figure 
subplot(2,2,1)
plot(n_periods, updates_sinA(1,:), n_periods, sinAparams(1,:))
ylim([-1 1]);
xlabel('Time')
ylabel('Updated a1 for deterministic A')
title('ProPar Updates of a1')
legend('updates', 'true a1')

subplot(2,2,2)
plot(n_periods, updates_sinA(2,:), n_periods, sinAparams(2,:))
ylim([-1 1]);
xlabel('Time')
ylabel('Updated a2 for deterministic A')
title('ProPar Updates of a2')
legend('updates', 'true a2')

subplot(2,2,3)
plot(n_periods, updates_sinA(3,:), n_periods, sinAparams(3,:))
ylim([-1 1]);
xlabel('Time')
ylabel('Updated a3 for deterministic A')
title('ProPar Updates of a3')
legend('updates', 'true a3')

subplot(2,2,4)
plot(n_periods, updates_sinA(4,:), n_periods, sinAparams(4,:))
ylim([-1 1]);
xlabel('Time')
ylabel('Updated a4 for deterministic A')
title('ProPar Updates of a4')
legend('updates', 'true a4')









    





