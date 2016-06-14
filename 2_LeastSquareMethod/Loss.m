%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Author] Terry Taewoong Um (terry.t.um@gmail.com) %
% Adaptive Systems Lab., University of Waterloo     %
% https://www.facebook.com/terryum.io/              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Please leave the author information when you reuse the codes.

% [Input]
% x : input data which produced y_pred
% y_pred : predicted value
% y_true : true value
% opt : {crossEntropy, L2}
% W : weight (for weight penalization)
% lamda : weight decay parameter

% [Output]
% y_out : one-hot vector (or vector)

function [loss d_loss] = Loss(x, y_pred, y_true, opt, W, lamda)
    if nargin < 6
        lamda = 0.5;
        if nargin < 5
            W = -1;
            if nargin < 4
                opt = 'crossEntropy';
            end
        end
    end
    
    loss = 0;       d_loss = 0;
    [nData, nOut] = size(y_pred);
    switch opt
        case 'crossEntropy'
        case 'L2'
            for ii=1:nData
                loss = loss + (y_true(ii,:)-y_pred(ii,:))*(y_true(ii,:)-y_pred(ii,:))';
                d_loss = d_loss + x(ii,:)'*(y_true(ii,:)-y_pred(ii,:));
            end
            loss = loss/nData;
            d_loss = d_loss/nData;
    end

    if W ~= -1
        loss_reg = sum(sum(W'*W));
    else
        loss_reg = 0;
    end
       
    loss = loss + lamda*loss_reg;
end