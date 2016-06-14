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

function [err_train, err_test, loss] = LeastSquareMethod_ND(x_train, y_train, x_test, y_test, nEpoch, stepsize, minibatch) 
    
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
    % Plot the data and result line
    [val idx1] = min(x_train(:,1));         [val idx2] = max(x_train(:,1));
    x_left = x_train(idx1,:);               x_right = x_train(idx2,:);
    y_left = [x_left 1]*W;                  y_right = [x_right 1]*W;
    fig_y1 = scatter3(x_train(:,1), x_train(:,2), y_train(:,1), 'b.');  hold on;
    fig_y2 = scatter3(x_train(:,1), x_train(:,2), y_train(:,2), 'k.');  hold on;
    fig_line1 = plot3([x_left(1,1); x_right(1,1)], [x_left(1,2); x_right(1,2)], [y_left(1,1); y_right(1,1)], 'r-', 'LineWidth', 3); hold on; 
    fig_line2 = plot3([x_left(1,1); x_right(1,1)], [x_left(1,2); x_right(1,2)], [y_left(1,2); y_right(1,2)], 'r-', 'LineWidth', 3); hold off;
    legend([fig_y1, fig_y2], 'y1', 'y2'); 
    xlabel('x1');   ylabel('x2');   zlabel('y1 & y2'); 

    % Show the errors     
    str_err = ['Train/Test error = ', num2str(err_train), ' / ', num2str(err_test)];
    x_pos = min(x_train(:,1))+ 0.1*abs(min(x_train(:,1)));   
    y_pos = min(x_train(:,2))+ 0.1*abs(min(x_train(:,2)));        
    z_pos = max(max(y_train));
    text(x_pos, y_pos, z_pos, str_err);   

   
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
    % Plot the data and result line    
    figure();
    [val idx1] = min(x_train(:,1));         [val idx2] = max(x_train(:,1));
    x_left = x_train(idx1,:);               x_right = x_train(idx2,:);
    y_left = [x_left 1]*W;                  y_right = [x_right 1]*W;
    fig_y1 = scatter3(x_train(:,1), x_train(:,2), y_train(:,1), 'b.');  hold on;
    fig_y2 = scatter3(x_train(:,1), x_train(:,2), y_train(:,2), 'k.');  hold on;
    fig_line1 = plot3([x_left(1,1); x_right(1,1)], [x_left(1,2); x_right(1,2)], [y_left(1,1); y_right(1,1)], 'r-', 'LineWidth', 3); hold on; 
    fig_line2 = plot3([x_left(1,1); x_right(1,1)], [x_left(1,2); x_right(1,2)], [y_left(1,2); y_right(1,2)], 'r-', 'LineWidth', 3); hold off;
    legend([fig_y1, fig_y2], 'y1', 'y2'); 
    xlabel('x1');   ylabel('x2');   zlabel('y1 & y2');

    % Show the errors     
    str_err = ['Train/Test error = ', num2str(err_train), ' / ', num2str(err_test)];
    x_pos = min(x_train(:,1))+ 0.1*abs(min(x_train(:,1)));   
    y_pos = min(x_train(:,2))+ 0.1*abs(min(x_train(:,2)));        
    z_pos = max(max(y_train));
    text(x_pos, y_pos, z_pos, str_err);   
    
    figure();
    fig_train = plot(err_log_train);    hold on;
    fig_test = plot(err_log_test);      hold off;
    xlabel('Epoch');    ylabel('MSE');
    legend([fig_train, fig_test], 'Train error', 'Test error');
    title('Error Log');

end

