function agent = CreateTD3Agent(numObs, obsInfo, numAct, actInfo, Ts)
%% Create Critic1 Network
statePath1 = [
    featureInputLayer(numObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(200,'Name','CriticStateFC2')];

actionPath1 = [
    featureInputLayer(1,'Normalization','none','Name','action')
    fullyConnectedLayer(200,'Name','CriticActionFC1','BiasLearnRateFactor',0)];

commonPath1 = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork1 = layerGraph(statePath1);
criticNetwork1 = addLayers(criticNetwork1,actionPath1);
criticNetwork1 = addLayers(criticNetwork1,commonPath1);
    
criticNetwork1 = connectLayers(criticNetwork1,'CriticStateFC2','add/in1');
criticNetwork1 = connectLayers(criticNetwork1,'CriticActionFC1','add/in2');

%% Create Actor Network
actorNetwork = [
    featureInputLayer(numObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(200,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(numAct,'Name','ActorFC3')
    tanhLayer('Name','ActorTanh1')
    scalingLayer('Name','ActorScaling','Scale',max(actInfo.UpperLimit))];
%% Create Critic2 Network
statePath2 = [
    featureInputLayer(numObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(200,'Name','CriticStateFC2')];

actionPath2 = [
    featureInputLayer(1,'Normalization','none','Name','action')
    fullyConnectedLayer(200,'Name','CriticActionFC1','BiasLearnRateFactor',0)];

commonPath2 = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork2 = layerGraph(statePath2);
criticNetwork2 = addLayers(criticNetwork2,actionPath2);
criticNetwork2 = addLayers(criticNetwork2,commonPath2);
    
criticNetwork2 = connectLayers(criticNetwork2,'CriticStateFC2','add/in1');
criticNetwork2 = connectLayers(criticNetwork2,'CriticActionFC1','add/in2');
%% オプション設定
if parallel.gpu.GPUDevice.isAvailable
    device = 'cpu';
else
    device = 'cpu';
end
% criticOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,... 
                                        'GradientThreshold',1,'L2RegularizationFactor',2e-4,'UseDevice',device);
% actorOptions = rlRepresentationOptions('LearnRate',5e-04,'GradientThreshold',1);                           
actorOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',5e-04,...
                                       'GradientThreshold',1,'L2RegularizationFactor',1e-5,'UseDevice',device);

%% optionを基にネットワーク定義（何をするネットワークなのか指定）Q値近似？or方策近似？
% options
critic1 = rlQValueRepresentation(criticNetwork1,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
critic2 = rlQValueRepresentation(criticNetwork2,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
actor  = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'ActorScaling'},actorOptions);

%% TD3 agent オプション設定（二種類の書き方があるのかな？）
agentOptions = rlTD3AgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.MiniBatchSize = 256;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.PolicyUpdateFrequency = 2; %policy update frequency
agentOptions.TargetSmoothFactor = 5e-3;
agentOptions.TargetPolicySmoothModel.StandardDeviation = 0.2; % target policy noise
agentOptions.TargetPolicySmoothModel.LowerLimit = -0.5;
agentOptions.TargetPolicySmoothModel.UpperLimit = 0.5;
agentOptions.ExplorationModel = rl.option.OrnsteinUhlenbeckActionNoise; % set up OU noise as exploration noise (default is Gaussian for rlTD3AgentOptions)
agentOptions.ExplorationModel.MeanAttractionConstant = 1;
agentOptions.ExplorationModel.StandardDeviation = 0.1;
%% Create agent using specified actor representation, critic representation and agent options
agent = rlTD3Agent(actor, [critic1,critic2], agentOptions);
