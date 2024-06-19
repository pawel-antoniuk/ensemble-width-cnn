% Resample and save routine
function HRTF = resampleAndSave(HRTF, params)
    fprintf('[%s][%s] Resampling from %d Hz to %d Hz\n', ...
        HRTF.HRTFGroup, HRTF.Name, ...
        HRTF.SOFA.Data.SamplingRate, ...
        params.RecordingsExpectedFs);

    HRTF.SOFA = SOFAresample(HRTF.SOFA, params.RecordingsExpectedFs);

    resampledSOFAdir = fullfile(params.HRTFBaseDir, ...
        ['_resampled_' num2str(params.RecordingsExpectedFs)], ...
        HRTF.HRTFGroup);
    resampledSOFAfilename = ['_resampled_' ...
        num2str(params.RecordingsExpectedFs) '_' HRTF.Name];

    if ~exist(resampledSOFAdir, 'dir')
        mkdir(resampledSOFAdir);
    end

    fullSOFAfilename = fullfile(resampledSOFAdir, resampledSOFAfilename);
    HRTF.SOFA = SOFAsave(fullSOFAfilename, HRTF.SOFA, 0);
end