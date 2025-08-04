function yout = thresh(x,y,psi,delta,idx)

% =========================== !! IMPORTANT !! ===========================
% 'BICYCLE_SAFETY_NN.M' must be executed prior to running this model,
% so that the cell array 'finalNets' is in the MATLAB workspace.
% =======================================================================



    yout=0; % THIS IS NECESSARY: it forces MATLAB to treat the output as a scalar double

    % Compute xi from the supplied states:
%     xi = atan2(y,x)-psi;
%     wrappedXi = xi - fix(xi/(2*pi))*(2*pi);
%     % Correct angles with abs() larger than pi:
%     if wrappedXi >= pi
%         wrappedXi = 2*pi - wrappedXi;
%     end
%     if wrappedXi <= -pi
%         wrappedXi = 2*pi + wrappedXi;
%     end
    
    % Tell simulink to go to the MATLAB interpreter for the function
    % 'predict':
    coder.extrinsic('predict')

    % Read the trained NNs from the MATLAB workspace:
    persistent trnet
    % Only read it in if it hasn't already been done:
    if ~ iscell(trnet)
        trnet = evalin('base','finalNets');
    end
    temp = predict(trnet{idx},[psi;delta]');
    yout = temp(1);
end