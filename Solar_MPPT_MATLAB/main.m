% MAIN.M Entrypoint for Solar MPPT MATLAB / Simulink Project
%
% Recreating the Reinforcement Learning-Based Solar Maximum Power Point
% Tracking System in MATLAB/Simulink with direct Python validation.
%
% Usage:
%   >> main

clear; clc; close all;

%% 1. Establish Project Environment and Paths
project_root = fileparts(mfilename('fullpath'));
fprintf('=================================================================\n');
fprintf('  REINFORCEMENT LEARNING SOLAR MPPT — MATLAB/SIMULINK PROJECT    \n');
fprintf('  Root Directory: %s\n', project_root);
fprintf('=================================================================\n\n');

% Add subdirectories to MATLAB search path
addpath(genpath(project_root));

%% 2. Load Centralized Configuration
cfg = project_config();
mppt_logger('INFO', 'Loaded centralized configuration for module: %s %s', ...
            cfg.pv.manufacturer, cfg.pv.model_name);
mppt_logger('INFO', 'Rated STC: Pmp=%.2f W, Vmp=%.2f V, Imp=%.2f A, Voc=%.2f V, Isc=%.2f A', ...
            cfg.pv.stc_p_mp, cfg.pv.stc_v_mp, cfg.pv.stc_i_mp, cfg.pv.stc_v_oc, cfg.pv.stc_i_sc);
mppt_logger('INFO', 'Converter: L=%.2f mH, Cin=%.1f uF, Cout=%.1f uF, Rload=%.1f Ohm, fsw=%.1f kHz', ...
            cfg.converter.inductance*1e3, cfg.converter.input_capacitance*1e6, ...
            cfg.converter.output_capacitance*1e6, cfg.converter.load_resistance, ...
            cfg.converter.switching_freq*1e-3);
mppt_logger('INFO', 'Multirate Timing: Plant dt=%.1f us, Control Ts=%.1f ms (Ratio: %d steps)', ...
            cfg.sim.plant_dt*1e6, cfg.sim.control_dt*1e3, cfg.sim.sub_steps);

%% 3. Execute Phase 0 Foundation Verification Tests
test_results = test_phase0_foundation();

if test_results.all_passed
    mppt_logger('INFO', 'Phase 0 Foundation verification SUCCESSFUL. Ready for Phase 1.');
else
    mppt_logger('ERROR', 'Phase 0 Foundation verification encountered failures. Check test log.');
end
