%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Author] Terry Taewoong Um (terry.t.um@gmail.com) %
% Adaptive Systems Lab., University of Waterloo     %
% https://www.facebook.com/terryum.io/              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Please leave the author information when you reuse the codes.

% [x_train, y_train, x_test, y_test] = GenerateData(opt, bShow, nData, sigma, test_ratio)

% opt : {gaussian(defalut), half_circle, full_circle, spiral}
% bShow : {0: Don't plot, 1(defalut): plot}
% nData : number of data (defalut: 1000)
% sigma : noise standard deviation (default: 1)
% test_ratio : ratio of test data (default: 0.2)

close all;  clearvars;

GenerateData();
GenerateData('half_circle');
GenerateData('full_circle', true, 800);
GenerateData('spiral', true, 1200, 1.2, 0.4);
