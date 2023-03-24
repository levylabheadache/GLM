function [basis, lags] = glmBasis(onsets, dilate, nframes, framerate, ...
                                  downsample, spacing, gausswidthsec, analog, dilate_analog)
%GLMBEHAVIOR Convert a series of onsets into a basis vector by first
%   converting to a series of delta functions and then convolving with a
%   gaussian.

    if nargin < 5 || isempty(downsample), downsample = 2; end
    if nargin < 6 || isempty(spacing), spacing = 2; end
    if nargin < 7 || isempty(gausswidthsec), gausswidthsec = 1; end
    if nargin < 8, analog = false; end
    if nargin < 9, dilate_analog = false; end

    onsets = double(onsets);
    
    % Calculate the convolution kernel
    % The default alpha for gausswin is 2.5
    % N points = round(2*alpha*sigma);
    % so, sigma = 2.4 for the example condition
    
    norm = gausswin(round(gausswidthsec*1.5*(framerate/downsample)));  % std of about 1 sec
    norm = norm./sum(norm);
    
    % Add a lags vector
    lags = [0];
    
    if analog && ~dilate_analog
        % No dilation necessary by definition, no conversion to vector
        % necessary, so just convolve and return
        basis = binvec(abs(onsets), downsample);
        basis = conv(basis, norm, 'same');
        return;
    elseif analog
        basis = binvec(abs(onsets), downsample);
    else
        % Make sure that onsets didn't run over the edge
        onsets = onsets(onsets > 0);
        onsets = onsets(onsets < nframes);

        % Generate single basis vector
        basis = zeros(1, nframes);
        basis(onsets) = 1;
        basis = binvec(basis, downsample);
        basis(basis > 0) = 1;
    end
    
    %% Account for situation where onsets need to be added
    
    nframes = length(basis);

    % Make it a basis matrix if necessary
    dilateframes = [round(dilate(1)*framerate/downsample), round(dilate(2)*framerate/downsample)];
    if dilateframes(1) == 0 && dilateframes(2) == 0
        basis = conv(basis, norm, 'same');
    else
        i = -spacing;
        while i >= dilateframes(1)
            newb = zeros(1, nframes);
            newb(1:end + i) = basis(1, -i + 1:end);
            basis = [basis; newb];
            lags = [lags i*downsample/framerate];
            i = i - spacing;
        end
        
        i = spacing;
        while i <= dilateframes(2)
            newb = zeros(1, nframes);
            newb(i + 1:end) = basis(1, 1:end - i);
            basis = [basis; newb];
            lags = [lags i*downsample/framerate];
            i = i + spacing;
        end
        
        for i = 1:size(basis, 1)
            basis(i, :) = conv(basis(i, :), norm, 'same');
        end
    end
end

