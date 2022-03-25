function agent = CreateDDPGAgent(numObs, obsInfo, numAct, actInfo, Ts)
%% Create Critic Network
statePath = [
    featureInputLayer(numObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(200,'Name','CriticStateFC2')];

actionPath = [
    featureInputLayer(1,'Normalization','none','Name','action')
    fullyConnectedLayer(200,'Name','CriticActionFC1','BiasLearnRateFactor',0)];

commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

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

%% オプション設定
if parallel.gpu.GPUDevice.isAvailable
    device = 'gpu';
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
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'ActorScaling'},actorOptions);
%% DDPG agent オプション設定（二種類の書き方があるのかな？）
% agentOptions = rlDDPGAgentOptions;
% agentOptions.SampleTime = Ts;
% agentOptions.DiscountFactor = 0.99;
% agentOptions.MiniBatchSize = 256;
% agentOptions.ExperienceBufferLength = 1e6;
% agentOptions.TargetSmoothFactor = 5e-3;
% agentOptions.NoiseOptions.MeanAttractionConstant = 1;
% agentOptions.NoiseOptions.StandardDeviation = 0.1;

agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'MiniBatchSize',128);
agentOptions.NoiseOptions.StandardDeviation = 0.4;
agentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-5;
%% Create agent using specified actor representation, critic representation and agent options
agent = rlDDPGAgent(actor,critic,agentOptions);


