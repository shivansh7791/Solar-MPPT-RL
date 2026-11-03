function cfg = project_config()
% PROJECT_CONFIG Centralized system configuration for Solar MPPT MATLAB/Simulink.
%
% Defines exact physical, electrical, and control parameters matching the
% validated Python implementation for Canadian Solar CS6P-240P, DC-DC Boost
% Converter, Classical P&O MPPT, Soft Actor-Critic (SAC) RL agent, and
% multirate simulation timing.
%
% Output:
%   cfg - Structure containing all sub-system configurations.

    %% 1. PV Module Specifications (Canadian Solar CS6P-240P)
    cfg.pv.manufacturer      = 'Canadian Solar';
    cfg.pv.model_name        = 'CS6P-240P';
    cfg.pv.technology        = 'Polycrystalline Silicon (poly-Si)';
    cfg.pv.num_cells         = 60;  % Series cells (Ns)
    
    % Standard Test Conditions (STC: 1000 W/m^2, 25 deg C, AM 1.5)
    cfg.pv.stc_irradiance    = 1000.0;    % Reference irradiance [W/m^2]
    cfg.pv.stc_temperature   = 25.0;      % Reference temperature [deg C]
    cfg.pv.stc_p_mp          = 240.097;   % Maximum Power Point power [W]
    cfg.pv.stc_v_mp          = 29.900;    % MPP voltage [V]
    cfg.pv.stc_i_mp          = 8.030;     % MPP current [A]
    cfg.pv.stc_v_oc          = 37.000;    % Open-circuit voltage [V]
    cfg.pv.stc_i_sc          = 8.590;     % Short-circuit current [A]

    % Temperature Coefficients
    cfg.pv.alpha_sc          = 0.005472;   % Current temp coefficient [A/deg C]
    cfg.pv.beta_oc           = -0.135198;  % Voltage temp coefficient [V/deg C]
    cfg.pv.gamma_r           = -0.4538;    % Power temp coefficient [%/deg C]
    cfg.pv.t_noct            = 44.1;       % Nominal Operating Cell Temp [deg C]

    % De Soto / CEC 5-Parameter Empirical Values
    cfg.pv.a_ref             = 1.577654;       % Modified ideality factor (n*Ns*Vt) [V]
    cfg.pv.I_L_ref           = 8.599262;       % Photocurrent at STC [A]
    cfg.pv.I_o_ref           = 5.52853e-10;    % Dark saturation current at STC [A]
    cfg.pv.R_s               = 0.310448;       % Series resistance [Ohm]
    cfg.pv.R_sh_ref          = 287.92276;      % Shunt resistance at STC [Ohm]
    cfg.pv.Adjust            = 3.568598;       % Temperature coefficient adjust parameter [%]

    %% 2. Fundamental Physical Constants
    cfg.constants.k_B        = 1.380649e-23;      % Boltzmann constant [J/K]
    cfg.constants.q          = 1.602176634e-19;   % Elementary charge [C]
    cfg.constants.k_eV       = 8.617333262e-5;    % Boltzmann constant [eV/K]
    cfg.constants.T_ref_K    = 298.15;            % Reference temperature [K] (25 deg C)
    cfg.constants.E_g_ref    = 1.121;             % Bandgap energy of Silicon at Tref [eV]
    cfg.constants.dEgdT      = -0.0002677;        % Bandgap temp dependence [eV/K]

    %% 3. DC-DC Boost Converter Specifications
    cfg.converter.inductance          = 1.0e-3;     % Boost inductor L [H] (1.0 mH)
    cfg.converter.inductor_res        = 0.05;       % Inductor ESR r_L [Ohm] (50 mOhm)
    cfg.converter.input_capacitance   = 100.0e-6;   % Input capacitor C_in [F] (100 uF)
    cfg.converter.output_capacitance  = 220.0e-6;   % Output filter capacitor C_out [F] (220 uF)
    cfg.converter.load_resistance     = 20.0;       % Nominal load resistance R_load [Ohm]
    cfg.converter.switching_freq      = 50000.0;    % PWM switching frequency [Hz] (50 kHz)
    cfg.converter.duty_min            = 0.10;       % Minimum duty cycle clamping bound
    cfg.converter.duty_max            = 0.85;       % Maximum duty cycle clamping bound
    cfg.converter.duty_init           = 0.35;       % Initial converter duty cycle command

    % Phase 3 Baseline Converter Parameters (for direct Phase 3 comparison)
    cfg.converter.phase3_c_in         = 470.0e-6;   % C_pv [F] (470 uF)
    cfg.converter.phase3_c_out        = 470.0e-6;   % C_out [F] (470 uF)
    cfg.converter.phase3_r_load       = 10.0;       % R_load [Ohm] (10 Ohm)
    cfg.converter.phase3_f_sw         = 20000.0;    % f_sw [Hz] (20 kHz)

    %% 4. Classical Perturb & Observe (P&O) Controller Parameters
    cfg.po.step_size                  = 0.005;      % Fixed duty cycle perturbation step (delta D)
    cfg.po.sampling_period            = 0.005;      % MPPT execution control period T_s [s] (5 ms, 200 Hz)
    cfg.po.duty_min                   = 0.10;       % Duty clamping lower bound
    cfg.po.duty_max                   = 0.85;       % Duty clamping upper bound
    cfg.po.duty_init                  = 0.35;       % Initial duty cycle at t=0

    %% 5. Soft Actor-Critic (SAC) Reinforcement Learning Agent Specifications
    cfg.sac.observation_dim           = 8;          % Realistic non-privileged observation space
    cfg.sac.action_dim                = 1;          % Continuous duty increment a_k in [-1, +1]
    cfg.sac.max_duty_step             = 0.05;       % Max duty cycle increment: delta D = a_k * delta_D_max
    cfg.sac.sampling_period           = 0.005;      % Control sampling period T_s [s] (5 ms)
    
    % Observation Normalization Scales
    cfg.sac.norm.v_pv                 = 50.0;       % Voltage normalization base [V]
    cfg.sac.norm.i_pv                 = 10.0;       % Current normalization base [A]
    cfg.sac.norm.p_pv                 = 300.0;      % Power normalization base [W]
    cfg.sac.norm.v_out                = 100.0;      % Output voltage normalization base [V]
    cfg.sac.norm.i_l                  = 10.0;       % Inductor current normalization base [A]

    %% 6. Multirate Simulation Timing Parameters
    cfg.sim.plant_dt                  = 1.0e-5;     % Physical plant integration timestep [s] (10 us, 100 kHz)
    cfg.sim.control_dt                = 5.0e-3;     % Controller decision timestep [s] (5 ms, 200 Hz)
    cfg.sim.sub_steps                 = round(cfg.sim.control_dt / cfg.sim.plant_dt); % 500 plant steps / control step
    cfg.sim.default_duration          = 1.5;        % Nominal simulation time [s] (300 control steps)
    cfg.sim.seed                      = 42;         % Global random seed for reproducible profiles

    %% 7. Operational Safety and Numerical Solver Bounds
    cfg.solver.max_iterations         = 100;        % Max Newton-Raphson iterations
    cfg.solver.tolerance_current      = 1e-10;      % Current convergence tolerance [A]
    cfg.solver.num_curve_points       = 500;        % Standard I-V curve resolution points
    cfg.solver.exp_arg_max            = 80.0;       % Exponential ceiling to prevent float overflow

end
