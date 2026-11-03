function results = test_phase0_foundation()
% TEST_PHASE0_FOUNDATION Unit test suite for Phase 0 Project Foundation.
%
% Verifies project configuration, reference dataset availability, multirate
% timing parameters, and core utilities.
%
% Output:
%   results - Struct with test outcomes and pass/fail summary.

    fprintf('=====================================================\n');
    fprintf('  RUNNING PHASE 0 FOUNDATION TESTS (MATLAB)\n');
    fprintf('=====================================================\n');

    tests_run = 0;
    tests_passed = 0;
    results.details = {};

    % Helper to register tests
    function assert_condition(cond, test_name, error_msg)
        tests_run = tests_run + 1;
        if cond
            tests_passed = tests_passed + 1;
            fprintf('  [PASS] %s\n', test_name);
            results.details{end+1} = struct('name', test_name, 'status', 'PASS', 'message', '');
        else
            fprintf(2, '  [FAIL] %s: %s\n', test_name, error_msg);
            results.details{end+1} = struct('name', test_name, 'status', 'FAIL', 'message', error_msg);
        end
    end

    %% Test 1: Configuration Loading
    try
        cfg = project_config();
        assert_condition(isstruct(cfg), 'Config Loading', 'project_config did not return a struct');
    catch ME
        assert_condition(false, 'Config Loading', ME.message);
        results.tests_run = tests_run;
        results.tests_passed = tests_passed;
        results.all_passed = false;
        return;
    end

    %% Test 2: Mandatory Configuration Fields
    required_fields = {'pv', 'converter', 'po', 'sac', 'constants', 'sim', 'solver'};
    has_all_fields = true;
    for i = 1:length(required_fields)
        if ~isfield(cfg, required_fields{i})
            has_all_fields = false;
            break;
        end
    end
    assert_condition(has_all_fields, 'Mandatory Config Fields', 'Missing one or more required fields in config');

    %% Test 3: Canadian Solar CS6P-240P Nameplate Ratings Consistency
    p_calc = cfg.pv.stc_v_mp * cfg.pv.stc_i_mp;
    p_err_pct = abs(p_calc - cfg.pv.stc_p_mp) / cfg.pv.stc_p_mp * 100.0;
    assert_condition(p_err_pct < 0.1, 'CS6P-240P Power Rating Consistency', ...
                     sprintf('Vmp * Imp = %.3f W deviates by %.4f%% from Pmp = %.3f W', p_calc, p_err_pct, cfg.pv.stc_p_mp));

    %% Test 4: Physical Parameter Positivity
    params_positive = (cfg.pv.stc_v_oc > 0) && (cfg.pv.stc_i_sc > 0) && ...
                      (cfg.pv.a_ref > 0) && (cfg.pv.I_L_ref > 0) && ...
                      (cfg.pv.I_o_ref > 0) && (cfg.pv.R_s >= 0) && (cfg.pv.R_sh_ref > 0);
    assert_condition(params_positive, 'PV Parameter Positivity', 'One or more physical parameters are non-positive');

    %% Test 5: Converter CCM Parameter Positivity
    conv_positive = (cfg.converter.inductance > 0) && ...
                    (cfg.converter.output_capacitance > 0) && ...
                    (cfg.converter.load_resistance > 0) && ...
                    (cfg.converter.switching_freq > 0);
    assert_condition(conv_positive, 'Converter Parameter Positivity', 'Converter values must be strictly positive');

    %% Test 6: Multirate Timing Integer Ratio
    ratio = cfg.sim.control_dt / cfg.sim.plant_dt;
    is_exact_ratio = abs(ratio - round(ratio)) < 1e-9 && (round(ratio) == cfg.sim.sub_steps);
    assert_condition(is_exact_ratio, 'Multirate Timing Integer Ratio', ...
                     sprintf('Control dt / Plant dt (%.1f) does not match sub_steps (%d)', ratio, cfg.sim.sub_steps));

    %% Test 7: Python Reference CSV Files Availability
    ref_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'reference_data', 'python_outputs');
    ref_files = {'stc_reference.csv', 'stc_iv_curve.csv', 'irradiance_sweep_mpp.csv', 'temperature_sweep_mpp.csv'};
    all_refs_exist = true;
    for i = 1:length(ref_files)
        fpath = fullfile(ref_dir, ref_files{i});
        if ~exist(fpath, 'file')
            all_refs_exist = false;
            fprintf('    Missing reference file: %s\n', ref_files{i});
        end
    end
    assert_condition(all_refs_exist, 'Python Reference Files Existence', 'Missing reference CSV files');

    %% Test 8: Logger Utility Functionality
    try
        mppt_logger('INFO', 'Test log message for Phase 0 verification.');
        assert_condition(true, 'Logger Utility', '');
    catch ME
        assert_condition(false, 'Logger Utility', ME.message);
    end

    %% Test 9: Comparison Utility Accuracy
    test_vec_a = [10.0; 20.0; 30.0];
    test_vec_b = [10.001; 20.002; 29.998];
    cmp = compare_matrices(test_vec_a, test_vec_b, 0.05, 'TestVector');
    assert_condition(cmp.passed && cmp.max_pct_error < 0.02, 'Comparison Utility Validation', ...
                     'compare_matrices failed accuracy check');

    %% Summary
    results.tests_run = tests_run;
    results.tests_passed = tests_passed;
    results.all_passed = (tests_passed == tests_run);

    fprintf('-----------------------------------------------------\n');
    fprintf('  Phase 0 Test Results: %d/%d Passed (%.1f%%)\n', ...
            tests_passed, tests_run, (tests_passed/tests_run)*100.0);
    fprintf('=====================================================\n');
end
