

## demo3_LeastSquareMethod_Regression.m

```matlab
[x_train, y_train, x_test, y_test] = GenerateData('regress_quad', false);
[err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test);

[x_train, y_train, x_test, y_test] = GenerateData('regress_exp', false);
[err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test);

[x_train, y_train, x_test, y_test] = GenerateData('regress_xsinx', false);
[err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test);

[x_train, y_train, x_test, y_test] = GenerateData('regress_exp_xsinx', false);
[err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test);

[x_train, y_train, x_test, y_test] = GenerateData('multi_linear', false, 1200, 0.5);
[err_train, err_test, loss] = LeastSquareMethod_ND(x_train, y_train, x_test, y_test);
```
![demo2_LeastSquareMethod](https://github.com/terryum/Tutorial_MachineLearning_Matlab/blob/master/demo_images/demo3_LeastSquareMethod_Regression.png)

## LeastSquareMethod_2D.m

```matlab
function [err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test, nEpoch, stepsize, minibatch) 
```

### 0. Set hyperparameters for numerical approach

```matlab
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
```

### 1. Analytic solution   

```matlab        
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
    ...
```

### 2. Numerical solution  

```matlab 

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
   (...)
```
    
