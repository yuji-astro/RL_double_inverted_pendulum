parameter
%% モデル定義
mdl = 'rlCartPoleDoublePen';
agentblk = [mdl '/RL Agent'];
open_system(mdl)

%% 環境定義
rng(0);
numObs = 8; 
numAct = 1;
obsInfo=rlNumericSpec([numObs 1]);  
actInfo=rlNumericSpec([numAct 1],'LowerLimit',-20,'UpperLimit',20);

env = rlSimulinkEnv(mdl,agentblk,obsInfo,actInfo);
% env.ResetFcn = @(in) setVariable(in,'theta0',-pi + 2*pi*rand);
env.ResetFcn = @(in)localResetFcn(in);

%% 学習の準備
Ts = 0.1; %エージェントのサンプリング時間
Tf = 10; %シミュレーション時間

%% Train
%エージェント生成
%DDPG agent
% agent = CreateDDPGAgent(numObs, obsInfo, numAct, actInfo, Ts);
%TD3 agent
% agent = CreateTD3Agent(numObs, obsInfo, numAct, actInfo, Ts);
% load('pretrain.mat','agent')

%Trainオプション定義
maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'ScoreAveragingWindowLength',10,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',400,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',400);

%並列処理を使うときのオプション
trainOpts.UseParallel = false;
trainOpts.ParallelizationOptions.Mode = 'async';
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;
trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';

%Train
doTraining = false;

if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load('pretrain.mat','agent')
end

%% テスト
simOptions = rlSimulationOptions('MaxSteps',300);
experience = sim(env,agent,simOptions);