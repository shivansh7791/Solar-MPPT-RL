function metrics = compare_matrices(actual, reference, tol_pct, metric_name)
% COMPARE_MATRICES Rigorous numerical comparison utility between MATLAB and Python.
%
% Calculates:
%   - Maximum Absolute Error (MAE)
%   - Maximum Relative / Percentage Error (%)
%   - Root Mean Square Error (RMSE)
%   - Mean Absolute Error (MAE)
%   - Correlation coefficient (r)
%   - Boolean pass status against specified tolerance
%
% Inputs:
%   actual      - Calculated MATLAB numerical array/scalar
%   reference   - Reference Python numerical array/scalar
%   tol_pct     - Permissible percentage error threshold (e.g. 0.05%)
%   metric_name - Optional descriptive name for reporting
%
% Output:
%   metrics     - Struct containing calculated comparative error metrics.

    if nargin < 3 || isempty(tol_pct)
        tol_pct = 0.05; % Default 0.05%
    end
    if nargin < 4 || isempty(metric_name)
        metric_name = 'Metric';
    end

    act = actual(:);
    ref = reference(:);

    if length(act) ~= length(ref)
        error('compare_matrices:DimensionMismatch', ...
              'Length of actual (%d) does not match reference (%d).', length(act), length(ref));
    end

    diff_abs = abs(act - ref);
    
    % Avoid division by zero in relative error
    denom = abs(ref);
    denom(denom == 0) = 1.0;
    diff_pct = (diff_abs ./ denom) * 100.0;

    metrics.metric_name   = metric_name;
    metrics.max_abs_error = max(diff_abs);
    metrics.mean_abs_err  = mean(diff_abs);
    metrics.rmse          = sqrt(mean(diff_abs.^2));
    metrics.max_pct_error = max(diff_pct);
    metrics.mean_pct_err  = mean(diff_pct);
    metrics.tolerance_pct = tol_pct;
    metrics.passed        = (metrics.max_pct_error <= tol_pct);

    if length(act) > 1 && std(act) > 0 && std(ref) > 0
        r_mat = corrcoef(act, ref);
        metrics.correlation = r_mat(1, 2);
    else
        metrics.correlation = 1.0;
    end
end
