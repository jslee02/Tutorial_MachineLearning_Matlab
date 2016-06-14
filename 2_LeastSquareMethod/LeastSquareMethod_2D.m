%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Author] Terry Taewoong Um (terry.t.um@gmail.com) %
% Adaptive Systems Lab., University of Waterloo     %
% https://www.facebook.com/terryum.io/              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Please leave the author information when you reuse the codes.

% Input
% nEpoch : number of epochs (default: 5)
% stepsize : update rate (default: 0.001)
% minibatch : number of required data for every update (default: 50)

function [err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test, nEpoch, stepsize, minibatch) 
    
    [nTrain, nDim_In] = size(x_train);
    [nTest, nDim_Out] = size(y_test);

    %% 0. Set hyperparameters for numerical approach
    if nargin < 7
        minibatch = 50;             % number of data in each minibatch
        if nargin < 6
        	stepsize = 0.001;       % W_new = W_old - stepsize*d_loss*W_old 
            if nargin < 5
                nEpoch = 200;
            end
        end 
    end
    opt_loss = 'L2';                % loss function = square(sum((y_pred-y_true)^2))
    
    %% 1. Analytic solution   
        
    % 1-1. Calculate the weight
    % y = WX  ->  W = (X'X)^(-1)X'y
    X = [x_train ones(nTrain,1)];   % y = Wx+b -> y = W[x 1]
    W = (inv(X'*X))*X'*y_train;
   
    % 1-2. Evaluation (mean square error)
    X_train = [x_train ones(nTrain,1)]; 
    X_test = [x_test ones(nTest,1)]; 
    
    [err_train d_loss] = Loss(X_train, X_train*W, y_train, opt_loss);
    [err_test d_loss] = Loss(X_test, X_test*W, y_test, opt_loss);
    
    % 1-3. Plot the data
    figure();
    % Get two points for plotting the result line   
    x_left = floor(min(x_train));    x_right = ceil(max(x_train));
    y_left = [x_left 1]*W;          y_right = [x_right 1]*W;
    fig_train = scatter(x_train, y_train, 'b.');    hold all;
    fig_test = scatter(x_test, y_test, 'c.');       hold all;
    fig_line = plot([x_left; x_right], [y_left; y_right], 'r-');    hold off;
    legend([fig_train, fig_test, fig_line], 'Train data', 'Test data', 'Result');

    str_err = ['Train/Test error = ', num2str(err_train), ' / ', num2str(err_test)];
    x_pos = min(x_train)+ 0.1*abs(min(x_train));   y_pos = max(y_train);
    text(x_pos, y_pos, str_err);
 
    
    %% 2. Numerical solution 
    
    % 2-1. Initialize the weight
    W = randn(nDim_In+1, nDim_Out)*0.1;
    X_train = [x_train ones(nTrain,1)]; 
    X_test = [x_test ones(nTest,1)]; 
    Y_train = y_train;
    
    % 2-2. Train the weight
    W_log = zeros(nDim_In+1, nDim_Out, nEpoch);
    nIter = floor(nTrain/minibatch);
    err_log = zeros(nIter*nEpoch,1);
    
    nErrLog = 0;
    for kk = 1:nEpoch
        W_log(:,:,kk) = W;      % For visualizing prediction improvements
        idx_shuffle = randperm(nTrain);
        X_train = X_train(idx_shuffle,:);   Y_train = Y_train(idx_shuffle,:);
        
        for ii=1:nIter
            nIdx1 = (ii-1)*minibatch+1;         nIdx2 = ii*minibatch;
            [err d_loss] = Loss(X_train(nIdx1:nIdx2, :), X_train(nIdx1:nIdx2, :)*W, Y_train(nIdx1:nIdx2,:), opt_loss);
            W = W + stepsize*d_loss;
            nErrLog = nErrLog+1;
            
            [err_log_train(nErrLog,1) d_loss] = Loss(X_train, X_train*W, Y_train, opt_loss);
            [err_log_test(nErrLog,1) d_loss] = Loss(X_test, X_test*W, y_test, opt_loss);
        end
    end
  
    
    % 2-3. Evaluation (mean square error)
    [err_train d_loss] = Loss(X_train, X_train*W, Y_train, opt_loss);
    [err_test d_loss] = Loss(X_test, X_test*W, y_test, opt_loss);
    loss = err_train;
    
    
    % 2-4. Plot the data
    figure();
    fig_train = scatter(x_train, y_train, 'b.');    hold all;
    fig_test = scatter(x_test, y_test, 'c.');       hold all;

    % Get two points for plotting the result line     
    x_left = floor(min(x_train));
    x_right = ceil(max(x_train));
    y_left = [x_left 1]*W;
    y_right = [x_right 1]*W;

    for kk=1:9
        mm = (kk-1)*floor(nEpoch/10)+1;
        y_temp1 = [x_left 1]*W_log(:,:,mm);
        y_temp2 = [x_right 1]*W_log(:,:,mm);
        plot([x_left; x_right], [y_temp1; y_temp2], 'color', [1 0.8 0.8]);    hold all;
    end 
    % Plot the data and result line
    fig_line = plot([x_left; x_right], [y_left; y_right], 'r-');    hold all;

    hold off;
    legend([fig_train, fig_test, fig_line], 'Train data', 'Test data', 'Result');

    % Show the errors     
    str_err = ['Train/Test error = ', num2str(err_train), ' / ', num2str(err_test)];
    x_pos = min(x_train)+ 0.1*abs(min(x_train));   y_pos = max(y_train);
    text(x_pos, y_pos, str_err);
        
    figure();
    fig_train = plot(err_log_train);    hold on;
    fig_test = plot(err_log_test);      hold off;
    xlabel('Epoch');    ylabel('MSE');
    legend([fig_train, fig_test], 'Train error', 'Test error');
    title('Error Log');

end

