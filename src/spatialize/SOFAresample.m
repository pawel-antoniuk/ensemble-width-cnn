% Resample SOFA routine
function Obj = SOFAresample(Obj, targetFs)
    currentFs = Obj.Data.SamplingRate;

    if currentFs == targetFs
        return
    end

    % Based on HRTFsamplingRateConverter10.m (S. Zieliński)
    M = size(Obj.Data.IR,1); % Number of measurements
    N = size(Obj.Data.IR,3); % Length of measurements
    IR = Obj.Data.IR;
    IR2 = zeros(M, 2, round(ceil(targetFs / currentFs * N*3) / 3));

    for ii = 1:M
        ir = squeeze(IR(ii, :, :))';
        irx3 = [ir; ir; ir];
        irx3 = resample(irx3, targetFs, currentFs);
        N2 = round(length(irx3)/3);
        ir2 = irx3(N2+1:2*N2, :);
        IR2(ii, :, :) = ir2';
    end

    Obj.Data.IR = IR2;
    Obj.Data.SamplingRate = targetFs;
    Obj=SOFAupdateDimensions(Obj);
end