function Summ = SummarizeGLM(Result, Pred, Resp, Opts)

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
end
defRow = find(contains(Summ.lofo.name, 'Deform'));
locoRow = find(contains(Summ.lofo.name, 'Loco') | contains(Summ.lofo.name, 'Kine'));
if isempty(defRow) || isempty(locoRow), error('Need LOFO to contain deformation and locomotion categories'); end

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

% Save the results
if ~isempty(Opts.saveRoot)
    savePath = sprintf('%s%s_results.mat', Opts.saveRoot, Opts.name ); % sprintf('%s_results.mat', Opts.saveRoot );
    save( savePath, 'Pred', 'Resp', 'Opts', 'Result', 'Summ', '-v7.3' );
    fprintf('\nSaved %s   ', savePath);  
end

end