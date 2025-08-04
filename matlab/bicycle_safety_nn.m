close all
%% Paremeters and Definitions

% Bicycle parameters:
lr = 2; % total vehicle length = 2*lr = 0.46 
% NOTE: THIS IS THE ~MAX VEHICLE LENGTH FOR SUBSEQUENT PARAMETERS!!
%   To allow a longer vehicle you may:
%       Increase \sigma;
%       INcrease rBar; or 
%       INcrease deltaFMax
deltaFMax = pi/4; % maximum steering angle: pi/4 = 45 degrees
betaMax = atan(0.5*tan(deltaFMax));
vmax = 5; % Maximum linear velocity

% Barrier function parameters:
rBar = 2; % Mimum physical distance to origin
sigma = 0.4; % Scaling factor that determines minimum dist. to 
		  % origin at xi = +-pi (see definition of 'barrier' below)

% The barrier function h:
h = @(v,r,xi) ( ...
		(sigma.*cos(xi/2) + 1 - sigma)/rBar - 1./r ...
	);

% The Lie derivative of h along trajectories of the (radial) bicycle dynamics:
Lh = @(v,r,xi,beta) (...
		v*( ...
			(1./r.^2).*cos(xi - beta) + ...
			sigma.*sin(xi/2).*sin(xi - beta)./(2*rBar*r) + ...
			sigma.*sin(xi/2).*sin(beta)./(2*rBar*lr) ...
		)...
	);

% 'barrier' computes h(v,r,xi) == 0 as a funciton of xi; i.e. the actual
% **physical barrier**.
barrier = @(xi) ( ...
		rBar./(sigma.*cos(xi/2) + 1 - sigma) ...
	);

% (Relaxation) class K function for CBF (this is just a linear function,
% scaled to account for the maximum velocity):
alpha = @(x) ( ...
		3*vmax*sigma*x./(2*rBar*lr) ...
	);


%% Plotting code for verification purposes
figure;
[XI,BETA] = meshgrid(linspace(-pi,pi,200),linspace(-betaMax,betaMax,200));
surf(XI,BETA,Lh(2,barrier(XI),XI,BETA)+alpha(h(2,barrier(XI),XI)));
hold on;
surf(XI,BETA,0*XI)
hold off;

figure;
[XI,BETA] = meshgrid(linspace(-pi,pi,200),linspace(-betaMax,betaMax,200));
surf(XI,BETA,Lh(1,barrier(XI)+1,XI,BETA)+alpha(h(1,barrier(XI)+1,XI)));
hold on;
surf(XI,BETA,0*XI)
hold off;

%% Train ReLU approximation safety network
layers = [
	sequenceInputLayer(1)
	fullyConnectedLayer(20)
	reluLayer
	fullyConnectedLayer(20)
	reluLayer
	fullyConnectedLayer(20)
	reluLayer
	fullyConnectedLayer(20)
	reluLayer
    fullyConnectedLayer(20)
	reluLayer
    fullyConnectedLayer(20)
	reluLayer
    fullyConnectedLayer(20)
	reluLayer
	fullyConnectedLayer(1)
	regressionLayer
];

% Create training data:
meshPoints = 2000;
xi = linspace(-pi,pi,meshPoints);

% Find a radius after which all controls are admissible:
freeControlThreshold = 10;
for rr=0:0.1:5
	if Lh(vmax,barrier(pi)+rr,pi,-betaMax)+alpha(h(vmax,barrier(pi)+rr,pi)) >= 0
		freeControlThreshold = rr;
		break;
	end
end

% Radius thresholds:
%	Each element should be interpreted as a 'physical' barrier of the form:
%		barrier(xi) + threshold(index).
%	(0 and freeControlThreshold are required.)
radiusThresholds = [0 2.5 5 freeControlThreshold];

% Each column will contain the training data for a single ReLU network:
betaThresh = zeros(length(xi),length(radiusThresholds)-1);
betaThreshOrig = zeros(length(xi),length(radiusThresholds)-1);
repThresh = zeros(length(radiusThresholds)-1,1);

safetyConst = 0.01;
for ii = 1:length(radiusThresholds)-1
	% Find the first angle where there is a constraint on the control:
	startAngle = fzero( ...
		@(XI) (Lh(vmax,barrier(XI)+radiusThresholds(ii),XI,-betaMax)+alpha(h(vmax,barrier(XI)+radiusThresholds(ii),XI))), ...
		pi ...
	);
	for jj = 1:length(xi)
		if xi(jj) >= startAngle
			if repThresh(ii) == 0
				repThresh(ii) = jj;
			end
			betaThresh(jj,ii) = ...
				fzero( ...
					@(beta) ( ...
						Lh(vmax,barrier(xi(jj))+radiusThresholds(ii),xi(jj),beta) + ...
						alpha(h(vmax,barrier(xi(jj))+radiusThresholds(ii),xi(jj))) ...
					), ...
					-betaMax ...
				) + safetyConst;
            betaThreshOrig(jj,ii) = betaThresh(jj,ii) - safetyConst;
        else
            betaThresh(jj,ii) = -betaMax;
            betaThreshOrig(jj,ii) = -betaMax;
		end
    end
end

trainedNets = {};
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
	'Shuffle','every-epoch', ...
	'MiniBatchSize',2000, ...
    'Verbose',false, ... 
	'MaxEpochs',500, ...
	'Plots','none'); % 'training-progress'
for ii = 1:length(radiusThresholds)-1
	trainedNets{ii} = trainNetwork(xi,betaThresh(:,ii)',layers,options);
% 	trainedNets{ii} = trainNetwork(xi(repThresh(ii):length(xi)),betaThresh(repThresh(ii):length(xi),ii)',layers,options);
end

%% Add betaMax clipping to NN:

for ii = 1:length(radiusThresholds)-1
    % Add a clipping layer to force the NN output into [-betaMax,betaMax]
    trainedNets{ii} = assembleNetwork([ ...
        trainedNets{ii}.Layers(1:length(trainedNets{ii}.Layers)-1); ...
        fullyConnectedLayer(1, ...
            'Weights',[1], ...
            'Bias',[betaMax] ...
        ); ...
        reluLayer; ...
        fullyConnectedLayer(1, ...
            'Weights',[-1], ...
            'Bias',[betaMax + 0.5*(max(betaThreshOrig(:,ii)) + betaMax)] ...
        ); ...
        reluLayer; ...
        fullyConnectedLayer(1, ...
            'Weights',[-1], ...
            'Bias',[0.5*(max(betaThreshOrig(:,ii)) + betaMax)] ...
        ); ...
        trainedNets{ii}.Layers(length(trainedNets{ii}.Layers)) ...
    ]);
end


%% Plot ReLU approximation compared to actual \beta threshold

figure
legText = {};
hold on
for ii = 1:length(radiusThresholds)-1
    plot( ...
        xi,betaThreshOrig(:,ii),  ...
        xi,predict(trainedNets{ii},xi),  ...
        -xi,-fliplr(betaThreshOrig(:,ii)),  ...
        xi,-predict(trainedNets{ii},-xi) ...
    );
    legText{4*ii - 3} = strcat('Lh(v,\xi,r) + \alpha(h(v,\xi,r)) == 0 for r = barrier(\xi) + ',  string(radiusThresholds(ii)));
    legText{4*ii - 2} = strcat('ReLU Approximation for r = barrier(\xi) + ',string(radiusThresholds(ii)));
    legText{4*ii - 1} = strcat('Lh(v,\xi,r) + \alpha(h(v,\xi,r)) == 0 for r = barrier(\xi) + ',  string(radiusThresholds(ii)));
    legText{4*ii    } = strcat('ReLU Approximation for r = barrier(\xi) + ',string(radiusThresholds(ii)));
end
plot(xi,betaMax + 0*xi,'g')
legText{length(legText)+1} = '\beta_{max}';
hold off
xlabel('\xi (radians)')
ylabel('\beta = tan^{-1}( 1/2 \cdot tan({\delta}_f))')
legend(legText, 'Location','northwest')


%% Augment NN to get high/low value filtering:

finalNets = trainedNets;

for ii = 1:length(radiusThresholds)-1
    % Add a clipping layer to force the NN output into [-betaMax,betaMax]
    path1 = finalNets{ii}.Layers(2:length(finalNets{ii}.Layers)-1);
    for kk = 1:length(path1)
        path1(kk).Name = strcat(path1(kk).Name,'_XiPath');
    end
    path1(length(path1)).Name = 'XiPathOut';
    path1 = [fullyConnectedLayer(1,'Name','XiPathIn','Weights',[1 0],'Bias',[0]); path1];
    
    path2 = finalNets{ii}.Layers(2:length(finalNets{ii}.Layers)-1);
    for kk = 1:length(path2)
        path2(kk).Name = strcat(path2(kk).Name,'_NegXiPath');
    end
    path2(length(path2)).Name = 'NegXiPathOut';
    path2 = [fullyConnectedLayer(1,'Name','NegXiPathIn','Weights',[-1 0],'Bias',[0]); path2];
    lgraph = layerGraph(path1);
    lgraph = addLayers(lgraph,path2);
    lgraph = addLayers(lgraph,fullyConnectedLayer(1,'Name','BetaPathIn','Weights',[0 1],'Bias',[0]));
    lgraph = addLayers(lgraph,sequenceInputLayer(2,'Name','InputLayer'));
    lgraph = connectLayers(lgraph,'InputLayer','XiPathIn');
    lgraph = connectLayers(lgraph,'InputLayer','NegXiPathIn');
    lgraph = connectLayers(lgraph,'InputLayer','BetaPathIn');
    lgraph = addLayers(lgraph,concatenationLayer(1,3,'Name','ConcatenationLayer'));
    lgraph = connectLayers(lgraph,'XiPathOut','ConcatenationLayer/in1');
    lgraph = connectLayers(lgraph,'NegXiPathOut','ConcatenationLayer/in2');
    lgraph = connectLayers(lgraph,'BetaPathIn','ConcatenationLayer/in3');
    lgraph = addLayers(lgraph, [...
        fullyConnectedLayer(4, 'Name', 'PathDifferences1', ...
            'Weights', [-1 0 1; 1 0 0; 0 -1 0; 0 0 1], ...
            'Bias', [0;betaMax;betaMax;betaMax] ...
        ) ...
        reluLayer('Name','relu_Differences1') ...
        fullyConnectedLayer(4, 'Name', 'PathDifferences2', ...
            'Weights', [-1 -1 1 0; 0 1 0 0; 0 0 1 0; 0 0 0 1], ...
            'Bias', [0;0;0;0] ...
        ) ...
        reluLayer('Name','relu_Differences2') ...
        fullyConnectedLayer(2, 'Name', 'PathDifferences3', ...
            'Weights', [-1 0 1 0; 0 0 0 1], ...
            'Bias', [-betaMax;-betaMax] ...
        )
    ]);
    lgraph = connectLayers(lgraph,'ConcatenationLayer','PathDifferences1');
    lgraph = addLayers(lgraph,regressionLayer('Name','FinalOutputLayer'));
    lgraph = connectLayers(lgraph,'PathDifferences3','FinalOutputLayer');
    finalNets{ii} = assembleNetwork(lgraph);
    exportONNXNetwork(finalNets{ii},strcat(num2str(ii), '.onnx'))
end


%% Final output:

% Safety "control-filter" NNs are stored in finalNets
%
% For each threshold in radiusThresholds, there is one safety
% "control-filter" network in finalNets that is valid for *all* radii
% larger than the barrier plus the associated radius threshold.
%
% That is:
%
% finalNet{ii} is valid for all r >= barrier(\xi) + radiusThresholds(ii)
%
% This can be relaxed to:
%
% finalNet{ii} is valid for all r >= barrier(PI) + radiusThresholds(ii)
%
% (so that the threshold is independent of \xi).
%
% NOTE: there is no need to test the *lower* bound associated with
% finalNet{1} because the barrier will ensure that it remains applicable.






