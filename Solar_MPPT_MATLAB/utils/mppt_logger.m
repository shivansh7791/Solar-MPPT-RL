function mppt_logger(level, message, varargin)
% MPPT_LOGGER Formatted console logging utility for Solar MPPT MATLAB.
%
% Usage:
%   mppt_logger('INFO', 'System initialized successfully.');
%   mppt_logger('DEBUG', 'Iteration %d: error = %.4e', iter, err);
%   mppt_logger('WARNING', 'Approaching duty saturation.');
%   mppt_logger('ERROR', 'Convergence failed in solver.');

    % Format user message
    if ~isempty(varargin)
        msg_str = sprintf(message, varargin{:});
    else
        msg_str = message;
    end

    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
    level_str = upper(strtrim(level));

    switch level_str
        case 'ERROR'
            fprintf(2, '[%s] [ERROR]   %s\n', timestamp, msg_str);
        case 'WARNING'
            fprintf('[%s] [WARNING] %s\n', timestamp, msg_str);
        case 'DEBUG'
            fprintf('[%s] [DEBUG]   %s\n', timestamp, msg_str);
        otherwise
            fprintf('[%s] [INFO]    %s\n', timestamp, msg_str);
    end
end
