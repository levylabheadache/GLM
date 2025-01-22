function GLMresultFig = ViewGLM(Pred, Resp, Opts, Result, Summ) 

% Determine which responses to plot
if sum(isnan(Opts.rShow))
    Opts.rShow = Summ.rGood;
end
%if size(Opts.rShow)
%close all;
if ~isempty(Opts.rShow)
    GLMresultFig = figure('WindowState','maximized', 'color','w');
    leftOpt = {[0.02,0.04], [0.08,0.03], [0.04,0.02]};  % {[vert, horz], [bottom, top], [left, right] }
    rightOpt = {[0.04,0.04], [0.08,0.01], [0.04,0.01]}; 
    xAngle = 25;
    Nrow = Pred.N+1; 
    Ncol = 8; 
    spGrid = reshape( 1:Nrow*Ncol, Ncol, Nrow )';
    if ~isempty(Opts.figDir)
        figPath = sprintf('%s%s', Opts.figDir, Opts.name ); %figPath = sprintf('%s%s_glmFits.pdf', Opts.figDir, Opts.name );
        if exist(figPath,'file') 
            delete(figPath); 
        end
    else
        figPath = '';
    end
    if strcmpi(Opts.xVar, 'Time')
        T = (1/Opts.frameRate)*(0:size(Pred.data,1)-1)'/60;
        xLabelStr = 'Time (min)';
    else
        T = (1:size(Pred.data,1))';
        xLabelStr = 'Scan';
    end
    %colorSet = vertcat([0,0,0], distinguishable_colors(Pred.N+Pred.fam.N));  %getColorSet(Pred.N+Pred.fam.N)
    
    % Plot predictors (zero-shift only)
    for v = 1:Pred.N
        sp(v) = subtightplot(Nrow, Ncol, spGrid(v+1,1:Ncol-1), leftOpt{:});
        plot( T, Pred.data(:,v) ); hold on;
        
        if Pred.N == 1
            ylabel('Vessel', 'Interpreter','none');
        else
            ylabel(Pred.name{v}, 'Interpreter','none');
        end 

        xlim([-Inf,Inf]);
        if v < Pred.N
            set(gca,'TickDir','out', 'TickLength',[0.003,0], 'box','off', 'XtickLabel',[]);
        else
            set(gca,'TickDir','out', 'TickLength',[0.003,0], 'box','off');
        end
    end
    xlabel(xLabelStr);
    
    % Plot each response and model results
    for r = Opts.rShow %Summ.rFit'
        % Plot response data and models' predictions
        sp(Pred.N+1) = subtightplot(Nrow, Ncol, spGrid(1,1:Ncol-1), leftOpt{:}); cla;
        plot( T, Resp.data(:,r), 'c' );  % , 'LineWidth',2
        hold on;
        plot( T, Result(r).prediction, 'k', 'LineWidth',1 );
        plot( T, Result(r).lopo.prediction );
        %plot( T, Result(r).lofo.prediction );
        legend(['Data','Full', Summ.lofo.name])
        set(gca,'XtickLabel',[], 'TickDir','out', 'TickLength',[0.003,0]);
        box off;
        summaryString = sprintf('%s: Response %i. %2.2f dev exp.', Opts.name, r, Result(r).dev);
        for fam = 1:Pred.fam.N
            summaryString = strcat(summaryString, sprintf('   %s: %2.2f',   Summ.lofo.name{fam}, Result(r).lofo.dev(fam))); % Result(r).lofo.devFrac(fam)
        end
        title(summaryString, 'Interpreter','none');
        ylabel(Resp.name{r}, 'Interpreter','none');
        xlim([-Inf,Inf]);

        % Plot coefficients for all predictors/delays
        for v = 1:Pred.N
            subtightplot(Nrow, Ncol, spGrid(v+1,Ncol), rightOpt{:}); cla;
            plot( Opts.lags, Result(r).coeff(:,v), 'k' ); hold on; 
            plot( Opts.lags, Result(r).coeff(:,v), 'k.', 'MarkerSize',10 ); 
            ylabel('Coeff'); % ylabel( sprintf('%s coeff', Pred.name{v}), 'Interpreter','none');
            title( sprintf('LOPO: %2.1f', 100*(1-Result(r).lopo.devFrac(v))) ); % locoDiamDeform_result{x}(1).lopo.devFrac()
            axis square; 
            xLim = get(gca,'Xlim');
            line(xLim, [0,0], 'color','k');
            %if v == 1, title('Predictor leads response >', 'FontSize',8, 'HorizontalAlignment','left'); end
        end
        %title('Predictor leads response >', 'FontSize',8, 'HorizontalAlignment','center');
        xlabel('Response Lag (s)'); %xlabel('Lag (s)'); %  if v == Pred.N,  end 
        linkaxes(sp, 'x');
        % compare full-model deviance explained to that of the LOPO and LOFO models
        subtightplot(Nrow, Ncol, spGrid(1,Ncol), rightOpt{:});  % [0.01,0.01], [0.00,0.00], [0.00,0.00]
        cla; %to clear the axis
        line([0,0], [0, Pred.N+Pred.fam.N+1]+0.5, 'color','k' ); hold on;
        line(Opts.minDev*[1,1], [0, Pred.N+Pred.fam.N+1]+0.5, 'color','r', 'LineStyle','--' );
        plot( [Result(r).dev, Result(r).lopo.dev, Result(r).lofo.dev], 1:Pred.N+Pred.fam.N+1,  '.' );
        axis square;
        set(gca, 'Ytick',1:Pred.N+Pred.fam.N+1, 'YtickLabel',['All', Summ.lopo.name, Summ.lofo.name], 'TickLabelInterpreter','none', 'FontSize',10, 'TickDir','out'); % 
        xlim([-0.02, Inf]); ylim([0, Pred.N+Pred.fam.N+1]+0.5); 
        ylabel('Dev Expl');

        % save figure
%         if ~isempty(figPath)
%             fprintf('\nSaving %s', figPath);
%             saveas(GLMresultFig ,figPath);
%         end

        if ~isempty(figPath)
            fprintf('\nSaving %s', figPath);
            export_fig(figPath, '-pdf', '-painters','-q101', '-append', GLMresultFig); 
            pause(1);
        else
            pause;
        end
    end
end
end