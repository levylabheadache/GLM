function [Result, Summ, Opts, Pred, Resp] = GLMparallel(Pred, Resp, Opts)
%
tic;
savePath = sprintf('%s%s_results.mat', Opts.saveRoot, Opts.name );
if exist(savePath,'file') && Opts.load
    fprintf('\nLoading %s ... ', savePath );
    loadStruct = load( savePath, 'Pred', 'Resp', 'Opts', 'Result', 'Summ' ); 
    Pred = loadStruct.Pred;
    Resp = loadStruct.Resp;
    Opts = loadStruct.Opts;
    Result = loadStruct.Result;
    Summ = loadStruct.Summ;
else
    if Pred.fam.N > 0 && ~all([Pred.fam.col{:}] <= Pred.N) 
        error('Family columns exceed number of predictors');  
    end  % Check that families are well-defined

    glmnetOptions = glmnetSet(Opts);  % Set the options to use
        
    % Generate temporal basis shifts of each predictor variable
    Pred.TB = [];
    for r = flip(1:Pred.N)
        for z = flip(1:Opts.Nshift) 
            Pred.TB(:,z,r) = circshift( Pred.data(:,r), Opts.shiftFrame(z), 1 ); % shift the predictor variables
        end
    end
    Pred.TB = Pred.TB(Opts.maxShift+1:end-Opts.maxShift,:,:); % remove frames that wrap around during cirshift
    Pred.TB = reshape( Pred.TB, [], Pred.N*Opts.Nshift, 1 );
    TBcol = reshape( 1:Pred.N*Opts.Nshift, Opts.Nshift, Pred.N )';
    Pred.data = Pred.data(Opts.maxShift+1:end-Opts.maxShift,:); % remove frames that wrap around during cirshift
    Resp.data = Resp.data(Opts.maxShift+1:end-Opts.maxShift,:); % remove frames that wrap around during cirshift

    % Define training/testing data subsets and (optionally) shuffles
    Nfull = size(Pred.TB, 1);
    trainWave = square( (2*pi*Opts.Ncycle)*(0:Nfull-1)/(Nfull-1), 100*Opts.trainFrac ); %square conflicts with an OASIS function, disable that package before running GLMparallel
    trainFrame = find( trainWave > 0 );
    testFrame = find( trainWave < 0 );
    %plot( trainWave ); xlabel('Frame'); set(gca,'Ytick',[0,1], 'box','off'); ylim([-1.01, 1.01] );
    Result = repmat(struct('obj',[], 'const',NaN, 'coeff',nan(Opts.Nshift,Pred.N), 'dev',NaN, 'prediction',[], 'peakCoeff',nan(1,Pred.N), 'peakShift',nan(1,Pred.N), 'lopo',[], 'lofo',[]), Resp.N, 1);
    fprintf('\nStarting parfor loop - %i scans, %i predictors,  %i response variables', size(Pred.TB, 1), size(Pred.TB, 2)  , Resp.N ); 
    fix(clock)
    tic 
    parfor r = 1:Resp.N % 
        %fprintf('\nr = %i', r);
        Result(r).lopo = struct('const',nan(1,Pred.N), 'coeff',nan(Opts.Nshift,Pred.N,Pred.N), 'obj',[], 'prediction',nan(Nfull,Pred.N), 'dev',nan(1,Pred.N), 'devFrac',nan(1,Pred.N) );
        Result(r).lofo = struct('const',nan(1,Pred.fam.N), 'coeff',nan(Opts.Nshift,Pred.N,Pred.fam.N), 'obj',[], 'prediction',nan(Nfull,Pred.fam.N), 'dev',nan(1,Pred.fam.N), 'devFrac',nan(1,Pred.fam.N) );
        if ~all(Resp.data(trainFrame,r)==0)
            Result(r).obj = cvglmnet( Pred.TB(trainFrame,:), Resp.data(trainFrame,r), Opts.distribution, glmnetOptions, 'deviance'); % , [], [], false, false, true
            tempCoeff = cvglmnetCoef(Result(r).obj);           
            Result(r).const = tempCoeff(1);
            Result(r).coeff = tempCoeff(2:end);
            Result(r).coeff = reshape( Result(r).coeff, Opts.Nshift, Pred.N );
            for p = find(~all(Result(r).coeff == 0, 1)) % flip(1:Pred.N)
                [~, tempPeakInd] = max( abs( Result(r).coeff(:,p) ), [], 1 );
                Result(r).peakCoeff(p) = Result(r).coeff(tempPeakInd,p);
                Result(r).peakShift(p) = Opts.lags(tempPeakInd);
            end
            Result(r).prediction = cvglmnetPredict(Result(r).obj, Pred.TB, [], 'response');
            [Result(r).dev, ~, ~] = GetDeviance(Resp.data(testFrame,r)', Result(r).prediction(testFrame)', nanmean(Resp.data(trainFrame,r)), Opts.distribution); % compare glm predictions to data
            % Leave one predictor out
            if Opts.lopo && Pred.N > 1
                for p = flip(1:Pred.N)
                    lopoTB = Pred.TB; lopoTB(:,TBcol(p,:)) = 0;
                    Result(r).lopo.obj{p} = cvglmnet( lopoTB(trainFrame,:), Resp.data(trainFrame,r), Opts.distribution, glmnetOptions, 'deviance'); % , [], [], false, false, true
                    tempCoeff = cvglmnetCoef(Result(r).lopo.obj{p});           
                    Result(r).lopo.const(p) = tempCoeff(1);
                    Result(r).lopo.coeff(:,:,p) = reshape( tempCoeff(2:end), Opts.Nshift, Pred.N );
                    Result(r).lopo.prediction(:,p) = cvglmnetPredict(Result(r).lopo.obj{p}, Pred.TB, [], 'response');
                    [Result(r).lopo.dev(p), ~, ~] = GetDeviance(Resp.data(testFrame,r)', Result(r).lopo.prediction(testFrame,p)', nanmean(Resp.data(trainFrame,r)), Opts.distribution);
                end
                Result(r).lopo.devFrac = Result(r).lopo.dev/Result(r).dev;
            end
            % Leave one family of predictors out
            if Pred.fam.N > 0
                for f = flip(1:Pred.fam.N)
                    famTBcols = reshape( TBcol(Pred.fam.col{f},:), 1, [] );
                    lofoTB = Pred.TB; lofoTB(:,famTBcols) = 0;
                    Result(r).lofo.obj{f} = cvglmnet( lofoTB(trainFrame,:), Resp.data(trainFrame,r), Opts.distribution, glmnetOptions, 'deviance'); % , [], [], false, false, true
                    tempCoeff = cvglmnetCoef(Result(r).lofo.obj{f});           
                    Result(r).lofo.const(f) = tempCoeff(1);
                    Result(r).lofo.coeff(:,:,f) = reshape( tempCoeff(2:end), Opts.Nshift, Pred.N );
                    Result(r).lofo.prediction(:,f) = cvglmnetPredict(Result(r).lofo.obj{f}, Pred.TB, [], 'response');
                    [Result(r).lofo.dev(f), ~, ~] = GetDeviance(Resp.data(testFrame,r)', Result(r).lofo.prediction(testFrame,f)', nanmean(Resp.data(trainFrame,r)), Opts.distribution);
                end
                Result(r).lofo.devFrac = Result(r).lofo.dev/Result(r).dev; 
            end
        end
    end
    toc
    
    fprintf('\nParfor loop completed.' ); fix(clock)
    % Summarize GLM results
    Summ.dev = [Result.dev];
    Summ.rGood = find(Summ.dev >= Opts.minDev); %   & [allTemp.p] < Opts.maxP
    Summ.Ngood = numel(Summ.rGood);
    Summ.peakCoeff = nan(Resp.N, Pred.N); 
    Summ.peakCoeff(Summ.rGood,:) = vertcat(Result(Summ.rGood).peakCoeff);
    Summ.peakLag = nan(Resp.N, Pred.N);
    Summ.peakLag(Summ.rGood,:) = vertcat(Result(Summ.rGood).peakShift);
    Summ.lopo = struct('dev',[], 'devFrac',[], 'rDependent',[], 'Ndependent',[], 'name',[]);
    Summ.lofo = struct('dev',[], 'devFrac',[], 'rDependent',[], 'Ndependent',[], 'name',[]);
    
    % How does dropping one predictor affect the model?
    if Opts.lopo && Pred.N > 1
        lopoTemp = [Result.lopo]; 
        Summ.lopo.dev = vertcat( lopoTemp.dev )'; % dropped predictor x response
        Summ.lopo.devFrac = vertcat( lopoTemp.devFrac )';
        % Which well-fit units depend on each variable for their explanatory value?
        goodUnits = Summ.dev >= Opts.minDev;
        dependentUnits = Summ.lopo.dev < Opts.minDev;
        for p = 1:Pred.N
            Summ.lopo.rDependent{p} = find(goodUnits & dependentUnits(p,:));
        end
        Summ.lopo.Ndependent = cellfun( @numel, Summ.lopo.rDependent );
        Summ.lopo.name = Pred.lopo.name;
    end
    
    % How does dropping one family of predictors affect the model?
    if Pred.fam.N > 1
        lofoTemp = [Result.lofo];
        Summ.lofo.dev = vertcat( lofoTemp.dev )'; % dropped family x response
        Summ.lofo.devFrac = vertcat( lofoTemp.devFrac )'; 
        % Which well-fit units depend on each family for their explanatory value?
        goodUnits = Summ.dev >= Opts.minDev;
        dependentUnits = Summ.lofo.dev < Opts.minDev;
        for f = 1:Pred.fam.N
            Summ.lofo.rDependent{f} = find(goodUnits & dependentUnits(f,:));
        end
        Summ.lofo.Ndependent = cellfun( @numel, Summ.lofo.rDependent );
        %Summ.lofo.name = Pred.fam.name;
        for f = flip(1:Pred.fam.N), Summ.lofo.name{f} = ['No ',Pred.fam.name{f}]; end
    end

    %{
    % Identify subtypes based on sensitivity to different families of predictors
    Summ.rIns = setdiff(1:Resp.N, Summ.rGood);
    [~,tempSort] = sort(Summ.dev(Summ.rIns), 'descend');
    Summ.rIns = Summ.rIns(tempSort);
    Summ.nIns = numel( Summ.rIns );

    Summ.rDeform = intersect(Summ.rGood, find(dependentUnits(defRow,:) & ~dependentUnits(locoRow,:) ) );  % which units are well fit only when deformation is included
    [~,tempSort] = sort(Summ.dev(Summ.rDeform), 'descend');
    Summ.rDeform = Summ.rDeform(tempSort);
    Summ.nDeform = numel( Summ.rDeform );

    Summ.rLoco = intersect(Summ.rGood, find(~dependentUnits(defRow,:) & dependentUnits(locoRow,:))); % setdiff(Summ.lofo.rDependent{2}, Summ.lofo.rDependent{1});
    [~,tempSort] = sort(Summ.dev(Summ.rLoco), 'descend');
    Summ.rLoco = Summ.rLoco(tempSort);
    Summ.nLoco = numel( Summ.rLoco );

    Summ.rMixed = setdiff(1:Resp.N, [Summ.rIns, Summ.rDeform, Summ.rLoco]  ); % intersect(Summ.lofo.rDependent{1}, Summ.lofo.rDependent{2});
    combinedDev = mean(Summ.lofo.dev([defRow,locoRow],:),1); %Summ.dev*(1 - abs(diff(Summ.lofo.dev([defRow,locoRow],:)))/abs(sum(Summ.lofo.dev([defRow,locoRow],:)))); %prod(Summ.lofo.dev,1);
    [~,tempSort] = sort(combinedDev(Summ.rMixed), 'descend'); % Summ.dev(Summ.rMixed)
    Summ.rMixed = Summ.rMixed(tempSort);
    Summ.nMixed = numel( Summ.rMixed );
    %}
    
    % Save the results
    if ~isempty(Opts.saveRoot)
        save( savePath, 'Pred', 'Resp', 'Opts', 'Result', 'Summ', '-v7.3' );
        fprintf('  Saved %s   ', savePath);  
    end
end
toc
end