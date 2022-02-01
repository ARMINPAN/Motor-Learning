%% Armin Panjehpour - Ardavan Kaviany - Jan,Feb 2021
%% Foundations of Neuroscience Project
% Processings on electrophysiology data from Motor Cortex of 
% Macaque monkey
% Task : Reach to Grasp
%% Dataset Information
% Data Acquisition using a 10*10 electrode array 
%% 
clc; clear; close all;
%% Part 0 - Load Dataset, Initialization -
%%%% Warning: dont run here - start running from section part.1 !!
Data = load('i140703-001_lfp-spikes.mat');

% spike trains of all 271 channels
spikes = Data.block.segments{1, 1}.spiketrains;  
spikeTimes = cell(271,1);

for i=1:length(spikes)
    spikeTimes(i) = mat2cell(spikes{1,i}.times,1,length(spikes{1,i}.times));
end
% events 
events = (Data.block.segments{1, 1}.events{1, 1});
eventsTimes = events.times;
eventsTrialedNamed = string(events.an_trial_event_labels);
eventsTrialedNamed = strrep(eventsTrialedNamed,' ','');
eventsTrialedLabeled = string(events.labels);
eventsMode = string(events.an_belongs_to_trialtype);
%% Warning: dont run here - start running from section part.1 !!
% at first we should remove bad trials by scrolling on labels, trials with
% wrong order of events will be removed
% a correct trial should have 8 events - 
% ["TS-ON";"WS-ON";"CUE-ON";"CUE-OFF";"GO-ON";"SR";"RW-ON";"STOP"] 
% "RW-ON" , "REP"s, 65381, 65386,
% 65440, 65504 are removed using the code below

remove = find(eventsTrialedNamed == "RW-OFF");
eventsTrialedNamed(remove) = [];
eventsTrialedLabeled(remove) = [];
eventsTimes(remove) = [];
eventsMode(remove) = [];

remove = contains(eventsTrialedNamed,"REP");
eventsTrialedNamed(find(remove == 1)) = [];
eventsTrialedLabeled(find(remove == 1)) = [];
eventsTimes(remove) = [];
eventsMode(remove) = [];

remove = find(eventsTrialedLabeled == "65381");
eventsTrialedNamed(remove) = [];
eventsTrialedLabeled(remove) = [];
eventsTimes(remove) = [];
eventsMode(remove) = [];

remove = find(eventsTrialedLabeled == "65386");
eventsTrialedNamed(remove) = [];
eventsTrialedLabeled(remove) = [];
eventsTimes(remove) = [];
eventsMode(remove) = [];

remove = find(eventsTrialedLabeled == "65440");
eventsTrialedNamed(remove) = [];
eventsTrialedLabeled(remove) = [];
eventsTimes(remove) = [];

remove = find(eventsTrialedLabeled == "65504");
eventsTrialedNamed(remove) = [];
eventsTrialedLabeled(remove) = [];
eventsTimes(remove) = [];
eventsMode(remove) = [];

remove = find(eventsTrialedLabeled == "65376");
eventsTrialedNamed(remove) = [];
eventsTrialedLabeled(remove) = [];
eventsTimes(remove) = [];
eventsMode(remove) = [];

%  now bad trials were identified and removed manually 
%% Warning: dont run here - start running from section part.1 !!
% now, in the most of the trials, monkey performs the task correctly and
% gets a reward but we`ve got some trials that monkey don`t get a reward
% which means it has failed to perform the task correctly

RewardingEventsTrialedNamed = [];
NoRewardingEventsTrialedNamed = [];
RewardingEventsTrialedLabeled = [];
NoRewardingEventsTrialedLabeled = [];
RewardingEventsTimes = [];
NoRewardingEventsTimes = [];
RewardingMode = [];
NoRewardingMode = [];

numberOfAllTrials = find(eventsTrialedNamed == "TS-ON");
for i=1:length(numberOfAllTrials)
    if(eventsTrialedNamed(numberOfAllTrials(i)+7) == "STOP") % rewarding
        RewardingEventsTrialedNamed = [RewardingEventsTrialedNamed,eventsTrialedNamed(numberOfAllTrials(i):numberOfAllTrials(i)+7)];
        RewardingEventsTrialedLabeled = [RewardingEventsTrialedLabeled, eventsTrialedLabeled(numberOfAllTrials(i):numberOfAllTrials(i)+7)];
        RewardingEventsTimes = [RewardingEventsTimes;eventsTimes(numberOfAllTrials(i):numberOfAllTrials(i)+7)];
        RewardingMode = [RewardingMode,eventsMode(numberOfAllTrials(i):numberOfAllTrials(i)+7)]; 
    elseif(eventsTrialedNamed(numberOfAllTrials(i)+7) == "TS-ON") % no rewarding
        NoRewardingEventsTrialedNamed = [NoRewardingEventsTrialedNamed,eventsTrialedNamed(numberOfAllTrials(i):numberOfAllTrials(i)+6)];        
        NoRewardingEventsTrialedLabeled = [NoRewardingEventsTrialedLabeled, eventsTrialedLabeled(numberOfAllTrials(i):numberOfAllTrials(i)+6)];
        NoRewardingEventsTimes = [NoRewardingEventsTimes;eventsTimes(numberOfAllTrials(i):numberOfAllTrials(i)+6)];
        NoRewardingMode = [NoRewardingMode,eventsMode(numberOfAllTrials(i):numberOfAllTrials(i)+6)]; 
    end
end

RewardingEventsTrialedNamed = RewardingEventsTrialedNamed.';
RewardingEventsTrialedLabeled = RewardingEventsTrialedLabeled.';
NoRewardingEventsTrialedNamed = NoRewardingEventsTrialedNamed.';
NoRewardingEventsTrialedLabeled = NoRewardingEventsTrialedLabeled.';
RewardingMode = RewardingMode.';
NoRewardingMode = NoRewardingMode.';

% 65296 start of task - 65360 of each trial - trials include the 400ms
% between TS-ON and WS-ON too, so in our definition, the begining of each
% trial is when TS-ON shows up


%% part.1 - Raster plot of spike patterns

figure; grid on; grid minor;
% events
titlesRewarding = ["TS-ON";"WS-ON";"CUE-ON";"CUE-OFF";"GO-ON";"SR";"RW-ON";"STOP"];
titlesNoRewarding = ["TS-ON";"WS-ON";"CUE-ON";"CUE-OFF";"GO-ON";"SR";"STOP"];

trialNumber = 142; % select which trial do you want to plot raster plot for

for i=1:length(RewardingEventsTimes(trialNumber,:))
    x1 = xline((RewardingEventsTimes(trialNumber,i)-RewardingEventsTimes(trialNumber,1))/30000,'--r',{titlesRewarding(i)},'Color','red');
end

for j=1:271
    pattern = spikeTimes{j,1};
    i = 1;
    while (pattern(i) < RewardingEventsTimes(trialNumber,8))
        if(RewardingEventsTimes(trialNumber,1) <= pattern(i))
        hold on;
        plot((pattern(i)-RewardingEventsTimes(trialNumber,1))/30000,j,'r.','color','black');
        end
        if(i == length(pattern))
            break;
        end
        i = i+1;
    end
end
title("Raster Plot || TrialNum." + trialNumber,'interpreter','latex');
ylabel('Neuron Number','interpreter','latex');
xlabel('Time(s)','interpreter','latex');
%% part.2 - PSTH
windowL = 300; % moving window length

duration = floor((RewardingEventsTimes(trialNumber,8)-RewardingEventsTimes(trialNumber,1))/windowL);

spikeT = zeros(271,(RewardingEventsTimes(trialNumber,8)-RewardingEventsTimes(trialNumber,1)));

% events
titlesRewarding = ["TS-ON";"WS-ON";"CUE-ON";"CUE-OFF";"GO-ON";"SR";"RW-ON";"STOP"];
titlesNoRewarding = ["TS-ON";"WS-ON";"CUE-ON";"CUE-OFF";"GO-ON";"SR";"STOP"];

figure; grid on; grid minor;
trialNumber = 15; % select which trial do you want to plot raster plot for
% 
for i=1:length(RewardingEventsTimes(trialNumber,:))
    x1 = xline((RewardingEventsTimes(trialNumber,i)-RewardingEventsTimes(trialNumber,1))/windowL,'--r',{titlesRewarding(i)},'Color','black');
end

for j=1:271
    pattern = spikeTimes{j,1};
    i = 1;
    k = 0;
    while (pattern(i) < RewardingEventsTimes(trialNumber,8))
        if(RewardingEventsTimes(trialNumber,1) <= pattern(i))
        k = k+1;
        spikeT(j,pattern(i)-RewardingEventsTimes(trialNumber,1)) = 1;
        end
        if(i == length(pattern))
            break;
        end
        i = i+1;
    end
end

%20 is the length of the window
% the duration of all trials are the same in here
summ = zeros(1,duration); % a vector to save number of spikes in each window of all trials
for ii=1:(duration)
    for i=1:271
        spikesTime = length(find(spikeT(i,((ii-1)*windowL+1):(ii)*windowL) == 1));
        % count number of spikes in the window
        summ(ii) = summ(ii) + spikesTime;
    end
end

hold on;
firerates = summ*30000/windowL;
bar(1:duration,firerates/271,'FaceColor','#A2142F','EdgeColor','#A2142F');
title("PETH|window length = 300 || 0.01s, TrialNum." + trialNumber,'interpreter','latex');
xlabel('window number','interpreter','latex');
ylabel('firing rate','interpreter','latex');
ylim([0 20]);
%% part.3 - ISI
ISI = cell(271,1);
for i=1:271
    targetNeuron = spikeTimes{i,1};
    ISIvector = [];
    ISIvector = diff(targetNeuron);
    ISI{i,1} = ISIvector;
end

% isi dist
figure;
targNeu = 207;
neuronISI = ISI{targNeu,1}/30000;
histfit(neuronISI,100,'gamma');
grid on; grid minor;
title("ISI Distribution of Neuron.Num"+targNeu,'interpreter','latex');
xlabel("ISI(s)",'interpreter','latex');
ylabel("Distribution of ISI",'interpreter','latex');
figure;
targNeu = 207;
neuronISI = ISI{targNeu,1}/30000;
histfit(neuronISI,100,'exponential');
grid on; grid minor;
title("ISI Distribution of Neuron.Num"+targNeu,'interpreter','latex');
xlabel("ISI(s)",'interpreter','latex');
ylabel("Distribution of ISI",'interpreter','latex');

%% part.4&5 - Statistical Analysis

%%%%%% part.2 code/ should run before this part of the code to calculate the
% firing rate

% as the raster plot and peth are showing, we can say the total activity of neurons 
% in motor cortex is increasing 

% so our null hypothesis is: SR-ONSET won`t increase the total activity of the
% area of motor cortex that the data is acquired

% we will calculate the standard mean of the neuron before SR-ONSET which
% monkey is held steady


windowL = 300; % moving window length
standardMean = zeros(1,length(RewardingEventsTimes));
SRmean = zeros(1,length(RewardingEventsTimes));
SRvar = zeros(1,length(RewardingEventsTimes)); % for fano factor
standardVar = zeros(1,length(RewardingEventsTimes)); % for fano factor

for kk=1:length(RewardingEventsTimes)
    durationn = floor((RewardingEventsTimes(kk,8)-RewardingEventsTimes(kk,1))/windowL);

    spikeTT = zeros(271,(RewardingEventsTimes(kk,8)-RewardingEventsTimes(kk,1)));
    trialNumber = kk
    for j=1:271
        pattern = spikeTimes{j,1};
        i = 1;
        k = 0;
        while (pattern(i) < RewardingEventsTimes(kk,8))
            if(RewardingEventsTimes(kk,1) < pattern(i))
            k = k+1;
            spikeTT(j,pattern(i)-RewardingEventsTimes(kk,1)) = 1;
            end
            if(i == length(pattern))
                break;
            end
            i = i+1;
        end
    end

    %20 is the length of the window
    % the duration of all trials are the same in here
    summ = zeros(length(RewardingEventsTimes),durationn); % a vector to save number of spikes in each window of all trials
    for ii=1:(durationn)
        for i=1:271
            spikesTime = length(find(spikeTT(i,((ii-1)*windowL+1):(ii)*windowL) == 1));
            % count number of spikes in the window
            summ(kk,ii) = summ(kk,ii) + spikesTime;
        end
    end

    firerates = summ(kk,:)*30000/windowL;

    standardMean(kk) = mean(firerates(1:(RewardingEventsTimes(kk,6)-RewardingEventsTimes(kk,1))/windowL)/271);
    standardVar(kk) = var(firerates(1:(RewardingEventsTimes(kk,6)-RewardingEventsTimes(kk,1))/windowL)/271);
    % now we calculate the mean of the firing rate of 271 neurons from SR-ONSET to 
    % RW-ONSET
    % this is for just one trial
    SRmean(kk) = mean(firerates(((RewardingEventsTimes(kk,6)-RewardingEventsTimes(kk,1))/windowL):((RewardingEventsTimes(kk,7)-RewardingEventsTimes(kk,1))/windowL))/271);
    SRvar(kk) = var(firerates(((RewardingEventsTimes(kk,6)-RewardingEventsTimes(kk,1))/windowL):((RewardingEventsTimes(kk,7)-RewardingEventsTimes(kk,1))/windowL))/271);
end


% two-sample t-test
df = length(SRmean)-1;
[h,p,ci,stats] = ttest(SRmean,standardMean);
tValue = stats.tstat;
CriticaltValue = tinv(0.95,df);

% % data distribution


figure;
histfit(standardMean,8);
hold on;

histfit(SRmean,8);
grid on; grid minor;



legend(["Baseline","baselineFit","SR-ONSET","SR-ONFit"]);
ylabel("Average Firing Rate on All 271 Neurons", 'interpreter', 'latex');
title("Firing Rate Distributions of Baseline and SR-ONSET", 'interpreter', 'latex');


%% part.6 - Fano Factor
% part 5 should be run before this part

% mean over all neurons and all trials in steady mode 
standardFano = standardVar./standardMean;
SRfano = SRvar./SRmean;


% are these fano factors different? - statistical analysis
% null hypothesis: the distributions are the same - paired sample t-test
% two-sample t-test
dfFano = length(SRfano) - 1;
[hFano,pFano,ciFano,statsFano] = ttest(standardFano,SRfano);
tValueFano = statsFano.tstat;
CriticaltValueFano = tinv(0.95,dfFano);

% data distribution
figure;
histfit(standardFano);
grid on; grid minor;

hold on;
histfit(SRfano);

legend(["Baseline","baselineFit","SR-ONSET","SR-ONFit"]);
ylabel("Average Fano Factor on All 271 Neurons", 'interpreter', 'latex');
title("Fano Factor Distributions of Baseline and SR-ONSET", 'interpreter', 'latex');


%% part.7 - trial segmentation based on the type of the task - |raster plot|

SGHFtaskRewarding = (find(RewardingMode(:,1) == "SGHF"));
SGLFtaskRewarding = (find(RewardingMode(:,1) == "SGLF"));
PGLFtaskRewarding = (find(RewardingMode(:,1) == "PGLF"));
PGHFtaskRewarding = (find(RewardingMode(:,1) == "PGHF"));

allTrials = {SGHFtaskRewarding;SGLFtaskRewarding;PGHFtaskRewarding;PGLFtaskRewarding};
% raster plot of 4 trials - each of a kind
figure; 
titlesRewarding = ["TS-ON";"WS-ON";"CUE-ON";"CUE-OFF";"GO-ON";"SR";"RW-ON";"STOP"];
titlesNoRewarding = ["TS-ON";"WS-ON";"CUE-ON";"CUE-OFF";"GO-ON";"SR";"STOP"];

for i=1:4
    % events
    subplot(2,2,i);
    targett = allTrials{i,1};
    trialNumber = targett(end-1) % select which trial do you want to plot raster plot for

    for ii=1:length(RewardingEventsTimes(trialNumber,:))
        x1 = xline((RewardingEventsTimes(trialNumber,ii)-RewardingEventsTimes(trialNumber,1))/30000,'--r',{titlesRewarding(ii)},'Color','red');
    end

    for j=1:271
        pattern = spikeTimes{j,1};
        ii = 1;
        while (pattern(ii) < RewardingEventsTimes(trialNumber,8))
            if(RewardingEventsTimes(trialNumber,1) <= pattern(ii))
            hold on;
            plot((pattern(ii)-RewardingEventsTimes(trialNumber,1))/30000,j,'r.','color','black');
            end
            if(ii == length(pattern))
                break;
            end
            ii = ii+1;
        end
        grid on; grid minor;
        title("Raster-Plot Mode" + RewardingMode(trialNumber),'interpreter','latex');
        xlabel("Time(s)",'interpreter','latex');
        ylabel("Neuron Number",'interpreter','latex');
    end
    
end

%% part.7 - trial segmentation based on the type of the task - |firing rate plot|
SGHFtaskRewarding = (find(RewardingMode(:,1) == "SGHF"));
SGLFtaskRewarding = (find(RewardingMode(:,1) == "SGLF"));
PGLFtaskRewarding = (find(RewardingMode(:,1) == "PGLF"));
PGHFtaskRewarding = (find(RewardingMode(:,1) == "PGHF"));

allTrials = {SGHFtaskRewarding;SGLFtaskRewarding;PGHFtaskRewarding;PGLFtaskRewarding};
% firing rate plot of 4 trials - each of a kind
spikeTime = zeros(length(allTrials),271); % a vector to save number of spikes in each window of all trials
figure;
B = zeros(4,271);
for kk=1:length(allTrials)
    for jjj=1:length(allTrials{kk,1})
        targett = allTrials{kk,1};
        trialNumber = targett(jjj) 

        durationn = ((RewardingEventsTimes(trialNumber,6)-RewardingEventsTimes(trialNumber,1):RewardingEventsTimes(kk,7)-RewardingEventsTimes(kk,1)));

        spikeTT = zeros(271,(RewardingEventsTimes(trialNumber,8)-RewardingEventsTimes(trialNumber,1)));
        for j=1:271
            pattern = spikeTimes{j,1};
            i = 1;
            k = 0;
            while (pattern(i) < RewardingEventsTimes(trialNumber,8))
                if(RewardingEventsTimes(trialNumber,1) < pattern(i))
                k = k+1;
                spikeTT(j,pattern(i)-RewardingEventsTimes(trialNumber,1)) = 1;
                end
                if(i == length(pattern))
                    break;
                end
                i = i+1;
            end
        end

        %20 is the length of the window
        % the duration of all trials are the same in here
        for i=1:271
            spikeTime(kk,i) = spikeTime(kk,i) + length(find(spikeTT(i,durationn) == 1));
        end
    end   
    firerates = (spikeTime(kk,:)*floor(30000/length(durationn)))/length(allTrials{kk,1});
    
    subplot(2,2,kk);
    bar(1:271,(spikeTime(kk,:)));
    grid on; grid minor;
    title("Mode" + RewardingMode(trialNumber),'interpreter','latex');
    ylabel("Averaged Firing Rate",'interpreter','latex');
    xlabel("Neuron Number",'interpreter','latex');
    ylim([0 3500])
end

%% %% part.7 - trial segmentation based on the type of the task - |PSTH|
% from the prev part we detected neuron num.17.113 responsible for encoding
SGHFtaskRewarding = (find(RewardingMode(:,1) == "SGHF"));
SGLFtaskRewarding = (find(RewardingMode(:,1) == "SGLF"));
PGLFtaskRewarding = (find(RewardingMode(:,1) == "PGLF"));
PGHFtaskRewarding = (find(RewardingMode(:,1) == "PGHF"));

bb = [17 113];
allTrials = {SGHFtaskRewarding;SGLFtaskRewarding;PGHFtaskRewarding;PGLFtaskRewarding};
% raster plot of 4 trials - each of a kind
Mode = ["SGHF","SGLF","PGHF","PGLF"];

            
duration = 2*30000;
windowL = 600;
for j=1:2
    spikeT = zeros(2,4,length(allTrials{kk,1}),60000);
    for kk=1:4
        for jjj=1:length(allTrials{kk,1})
            targett = allTrials{kk,1};
            trialNumber = targett(jjj); % select which trial do you want to plot raster plot for
            pattern = spikeTimes{bb(j),1};
            i = 1;
            while (pattern(i) < (RewardingEventsTimes(trialNumber,6)+1.5*30000))
                if((RewardingEventsTimes(trialNumber,6)-0.5*30000) <= pattern(i))
                    spikeT(j,kk,jjj,pattern(i)-RewardingEventsTimes(trialNumber,6)+0.5*30000) = 1;
                end
                if(i == length(pattern))
                    break;
                end
                i = i+1;
            end
        end
    end
    
    %20 is the length of the window
    % the duration of all trials are the same in here
%     targNeuron = reshape(mean(spikeT(j,:,:,:),3),[1 4 60000]);
    summ = zeros(2,4,length(allTrials{kk,1}),duration/windowL); % a vector to save number of spikes in each window of all trials
    for ii=1:4
        for i=1:length(allTrials{kk,1})
            for jjjj=1:duration/windowL
                spikesTime = length(find(spikeT(j,ii,i,((jjjj-1)*windowL+1):(jjjj)*windowL) == 1));
                % count number of spikes in the window
                summ(j,ii,i,jjjj) = summ(j,ii,i,jjjj) + spikesTime;
            end
        end
    end
    
    figure;
    for i=1:4
        su = reshape(mean(summ(j,i,:,:),3),[1 60000/windowL]);
        firerates = su*30000/windowL;
        plot((1:duration/(windowL))*0.02,firerates,'LineWidth',3);
        ylim([0 300]);
        hold on;
    title("Averaged Firing Rate - window length: 600 - Neuron" + bb(j),'interpreter','latex');
    ylabel("Averaged Firing Rate",'interpreter','latex');
    xlabel("Time(s)",'interpreter','latex');
    end
                    grid on; grid minor; 
                   x1 = xline(0.5*30000/windowL*0.02,'--r',{'SR'},'Color','red');
         legend(["SGHF","SGLF","PGHF","PGLF","SR-ONSET"]);
         
end
%% part.8 - LFP signal
numberOfChannels = 96;
LFPsignal = zeros(numberOfChannels, length(Data.block.segments{1, 1}.analogsignals{1, 1}.signal)); % 1-96 electrodes are just needed
for i=1:numberOfChannels
    LFPsignal(i,:) = Data.block.segments{1, 1}.analogsignals{1, 1}.signal.';
end

SGHFtaskRewarding = (find(RewardingMode(:,1) == "SGHF"));
SGLFtaskRewarding = (find(RewardingMode(:,1) == "SGLF"));
PGLFtaskRewarding = (find(RewardingMode(:,1) == "PGLF"));
PGHFtaskRewarding = (find(RewardingMode(:,1) == "PGHF"));

titlesRewarding = ["TS-ON";"WS-ON";"CUE-ON";"CUE-OFF";"GO-ON";"SR";"RW-ON";"STOP"];
titlesNoRewarding = ["TS-ON";"WS-ON";"CUE-ON";"CUE-OFF";"GO-ON";"SR";"STOP"];


allTrials = {SGHFtaskRewarding;SGLFtaskRewarding;PGHFtaskRewarding;PGLFtaskRewarding};


wt = cell(1,4);
for i=1:4
    figure; 
    targettt = allTrials{i,1};
    x = zeros(1,3*1000);
    for j=1:length(targettt)
        trialNumber = targettt(j);
        y = mean(LFPsignal(:,(floor(RewardingEventsTimes(trialNumber,6)/30000*1000)-1.5*1000+1):(floor(RewardingEventsTimes(trialNumber,6)/30000*1000)+1.5*1000)));
        x = x + y;
        %stft(x,'Window',kaiser(256,5),'OverlapLength',220,'FFTLength',512);
    end
	wt{1,i} = cwt(x,1000);
    wt_baselineNorm = wt{1,i};
    meann = mean(wt_baselineNorm(:,1:end/2),2);
    stdd = std(wt_baselineNorm(:,1:end/2),0,2);
    wt_baselineNorm = (wt_baselineNorm-meann)./stdd;
    wt{1,i} = wt_baselineNorm;
    cwt(icwt(wt_baselineNorm),1000)
    tts = xline(1.5,'--r',{'SR'},'Color','white');
    title("Magnitude Scalogram " + Mode(i));
end
%% functions goes here
