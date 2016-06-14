%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Author] Terry Taewoong Um (terry.t.um@gmail.com) %
% Adaptive Systems Lab., University of Waterloo     %
% https://www.facebook.com/terryum.io/              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Please leave the author information when you reuse the codes.

% [x_train, y_train, x_test, y_test] = GenerateData(opt, bShow, nData, sigma, test_ratio)

% opt : {class_gaussian(defalut), class_half_circle, class_full_circle, class_spiral,
%        regress_quad, regress_exp, regress_xsinx, regress_exp_xsinx, 
%        twolink_linear (Terry's Lie group toolbox required, 
%        [Link] https://github.com/terryum/Human-Robot-Motion-Simulator-based-on-Lie-Group)}
% bShow : {0: Don't plot, 1(defalut): plot}
% nData : number of data (defalut: 1000)
% sigma : noise standard deviation (default: 1)
% test_ratio : ratio of test data (default: 0.2)

close all;  clearvars;

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

% % Terry's Lie group toolbox required 
% [x_train, y_train, x_test, y_test] = GenerateData('multi_twolink', false, 2000, 0.1);
% [err_train, err_test, loss] = LeastSquareMethod_ND(x_train, y_train, x_test, y_test);

