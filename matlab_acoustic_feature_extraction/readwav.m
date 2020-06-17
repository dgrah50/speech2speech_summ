%{
    ACOUSTIC FEATURE EXTRACTOR SCRIPT.
    WAV file -> N-dimension feature vector
    The WAV file is analysed in 100ms frames
    For each 100ms frame a feature vector is generated containing the following information:
    The min, max, median, mean and range of the pitch (based on 10ms subframes)
    The min, max, median, mean and range of the energy (based on 10ms subframes)
    Mel Cepstral Coefficients + 1st and 2nd derivatives
%}

% acousticfeaturevec('osr1.wav')
% myDir = '/Users/dayangraham/Desktop/speech2speech_summ/speech_audios'; % gets directory
% myFiles = dir(fullfile(myDir,'*.wav'));
% 
% % dir(uigetdir)
% allFileNames = {myFiles(:).name};
% 
% for k = 1:length(allFileNames)
% 
%     filepath = strcat(myDir, allFileNames{k});
%     ret = acousticfeaturevec(filepath);
%     basename = erase(allFileNames{k}, ".wav")
%     save( sprintf(basename),'ret');
% end

acousticfeaturevec('/Users/dayangraham/Desktop/speech2speech_summ/speech_audios/NNTRAla40xk.wav')

function fv = acousticfeaturevec(wavfile)
    %%%%%%%%%%%%%%%%%%% PRELIMINARY SECTION AND SETUP %%%%%%%%%%%%%%%%%%%
    %Sample frequency
    fs = 8000;

    %We are analysing 100ms segments of the signal and assigning a prominence score to each segment
    binlength = 0.1;

    %Load wav
    [waveform] = v_readwav(wavfile, 'p', -1, 0);

    %%%%%%%%%%%%%%%%%%% VOICE ACTIVIY LEVEL FEATURE EXTRACTION %%%%%%%%%%%%%%%%%%%
    % Calculate approximate voice activity level from p.56 activity level
    %  [a, b, c, VAD] = v_activlev(waveform, 8000, 'n');
    [vs] = v_vadsohn(waveform, fs, 'nb');
    vad = vadframesplit(vs(:, 3));

    %%%%%%%%%%%%%%%%%%% PITCH FEATURE EXTRACTION %%%%%%%%%%%%%%%%%%%
    %pitch estimation algorithm - give analysis in 10ms frames
    [pitch_est] = v_fxpefac(waveform, fs, 0.01);

    % %calculates the min,max, median, mean and range of the pitch within a 100
    % ms frames
    [pitchstats] = pitchframesplit(pitch_est);

    %%%%%%%%%%%%%%%%%%% MELCEPT FEATURE EXTRACTION %%%%%%%%%%%%%%%%%%%
    melp = melcepstcalc(waveform,fs,binlength);

    %%%%%%%%%%%%%%%%%%% CONSTRUCTION OF FEATURE VECTOR %%%%%%%%%%%%%%%%%%%
     fv = constructfeaturevector(vad,pitchstats,melp);
     
     %%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%

function [melp] = melcepstcalc(signal, fs,binlength)
    melp = v_melcepst(signal, fs, 'E', 12, floor(3 * log(fs)), 1600);
    %{
    melp Inputs:
     s      speech signal
     fs  sample rate in Hz (default 11025)
     w   mode string (see below)
     nc  number of cepstral coefficients excluding 0'th coefficient [default 12]
     p   number of filters in v_filterbank [default: floor(3*log(fs)) =  approx 2.1 per ocatave]
     n   length of frame in samples [default power of 2 < (0.03*fs)]
     inc frame increment [default n/2]
     fl  low end of the lowest filter as a fraction of fs [default = 0]
     fh  high end of highest filter as a fraction of fs [default = 0.5]
    %}
    
    % Calculate first and second order derivatives of the melceps
    % Concatenate zeros to the front to adjust for length
    firstdiff = cat(1, zeros(1, 13), diff(melp));
    seconddiff = cat(1, zeros(2, 13), diff(melp,2));
    melp = horzcat(horzcat(melp, firstdiff), seconddiff);
end

function [pitchstats] = pitchframesplit(signal)
    % Number of groups
    nGroup = floor(length(signal)*0.1);
    % Number of element per each group, should 
    nElementPerGroup = 10;
    % Create a group index
    group = repelem((1:nGroup), nElementPerGroup);
    % Transpose signal to use splitapply
    signal = transpose(signal);
    % Stats of each group
    [lo, hi, med, avg, rng] = splitapply(@multiStat, signal(1:numel(group)), group);
    pitchstats = vertcat(lo, hi, med, avg, rng);
    pitchstats = transpose(pitchstats);
end

function [avg] = vadframesplit(signal)
    % Number of groups
    nGroup = floor(length(signal) * 0.1);
    % Number of element per each group, should
    nElementPerGroup = 10;
    % Create a group index
    group = repelem((1:nGroup), nElementPerGroup);
    % Transpose signal to use splitapply
    signal = transpose(signal);
    % Stats of each group
    [avg] = splitapply(@mean, signal(1:numel(group)), group);
    avg = transpose(avg);

end

function [lo, hi, med, avg, rng] = multiStat(x)
    lo = min(x);
    avg = mean(x);
    med = median(x);
    rng = range(x);
    hi = max(x);
end

function [fv] = constructfeaturevector(vad,pitchstats,melp) 
    %Truncate all feature vectors before concatenation
    minlen = min([length(vad),length(pitchstats),length(melp)]);
    vad = vad(1:minlen, :);
    pitchstats = pitchstats(1:minlen, :);
    melp =  melp(1:minlen, :);
    %Concatenate feature vectors
    assignin('base','vad',vad);
    assignin('base','pitchstats',pitchstats);
    assignin('base','melp',melp);
    fv = horzcat(horzcat(vad,pitchstats),melp);
end

end
