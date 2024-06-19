% Load HRTFs routine
function HRTFs = loadHRTFs(params)
    HRTFFilenames = dir(fullfile(params.HRTFBaseDir, '*', '*.sofa'));

    % HRTF struct definition
    HRTFs = struct('Id', [], ...
        'Name', [], ...
        'Folder', [], ...
        'HRTFGroup', [], ...
        'SOFA', [], ...
        'Position', [], ...
        'Distance', []);
    HRTFGroupData = containers.Map;

    for iHRTF = 1:length(HRTFFilenames)
        filename = HRTFFilenames(iHRTF);
        fullFilename = fullfile(filename.folder, filename.name);

        HRTFs(iHRTF) = loadHRTF(iHRTF, fullFilename, params);

        if HRTFs(iHRTF).SOFA.Data.SamplingRate ~= params.RecordingsExpectedFs
            [loadStatus,HRTFs(iHRTF)] = tryLoadResampledHRTF(iHRTF, ...
                HRTFs(iHRTF), params);
            if ~loadStatus
                resampleAndSave(HRTFs(iHRTF), params);
                [loadStatus,HRTFs(iHRTF)] = tryLoadResampledHRTF(iHRTF, ...
                    HRTFs(iHRTF), params);

                if ~loadStatus
                    error('Cannot find previously resampled HRTF');
                end
            end
        end

        ir =  HRTFs(iHRTF).SOFA.Data.IR;
        if size(ir, 3) > params.IRmax
            Nfadeout = params.FadeDuration*params.RecordingsExpectedFs;
            fade = [repelem(1,params.IRmax-Nfadeout), ...
                (cos(linspace(0,pi,Nfadeout))+1)/2];
            fade = reshape(fade,1,1,[]);
            ir = ir(:, :, 1:params.IRmax);
            HRTFs(iHRTF).SOFA.Data.IR = ir .* fade;
        end

        if ~isKey(HRTFGroupData, HRTFs(iHRTF).HRTFGroup)
            HRTFGroupData(HRTFs(iHRTF).HRTFGroup) = [];
        end

        HRTFGroupData(HRTFs(iHRTF).HRTFGroup) = [...
            HRTFGroupData(HRTFs(iHRTF).HRTFGroup) iHRTF];

        fprintf('[%s][%s] azimuth: [%d, %d]; elevation: [%d, %d]; distance: %d\n', ...
            HRTFs(iHRTF).HRTFGroup, ...
            HRTFs(iHRTF).Name, ...
            min(HRTFs(iHRTF).Position(:, 1)), ...
            max(HRTFs(iHRTF).Position(:, 1)), ...
            min(HRTFs(iHRTF).Position(:, 2)), ...
            max(HRTFs(iHRTF).Position(:, 2)), ...
            HRTFs(iHRTF).Distance);

        if HRTFs(iHRTF).SOFA.Data.SamplingRate ~= params.RecordingsExpectedFs
            error('[%s][%s] Resampling from %d Hz to %d Hz', ...
                HRTF.HRTFGroup, HRTF.Name, ...
                HRTF.SOFA.Data.SamplingRate, ...
                params.RecordingsExpectedFs);
        end
    end
end


% Try load resampled HRTF routine
function [loadStatus, HRTF] = tryLoadResampledHRTF(id, HRTF, params)
    resampledSOFAdir = fullfile(params.HRTFBaseDir, ...
        ['_resampled_' num2str(params.RecordingsExpectedFs)], ...
        HRTF.HRTFGroup);
    resampledSOFAfilename = ['_resampled_' ...
        num2str(params.RecordingsExpectedFs) '_' HRTF.Name];
    fullSOFAfilename = fullfile(resampledSOFAdir, resampledSOFAfilename);

    if ~exist(fullSOFAfilename, 'file')
        loadStatus = false;
    else
        loadStatus = true;
        HRTF = loadHRTF(id, fullSOFAfilename, params);
    end
end


% Load HRTF routine
function HRTF = loadHRTF(id, filename, params)
    listing = dir(filename);
    fullFilename = fullfile(listing.folder, listing.name);
    filenameParts = split(listing.folder, filesep);
    SOFA = SOFAload(fullFilename);
    APV = SOFAcalculateAPV(SOFA);

    HRTF.Id = id;
    HRTF.Name = listing.name;
    HRTF.Folder = listing.folder;
    HRTF.HRTFGroup = filenameParts{end};
    HRTF.SOFA = SOFA;
    HRTF.Position = APV(:, 1:2);
    HRTF.Distance = unique(HRTF.SOFA.SourcePosition(:, 3));

    if any(strcmp(HRTF.HRTFGroup, params.InverseAzimuthHRTFGroups))
        HRTF.Position = HRTF.Position * [-1 0; 0 1];
    end

    if mod(HRTF.SOFA.API.N, 2) ~= 0
        tmpIR = HRTF.SOFA.Data.IR(:, :, 1:end-1); % Remove last sample
        HRTF.SOFA.Data.IR = tmpIR;
        HRTF.SOFA.API.N = size(tmpIR, 3);
    end
end