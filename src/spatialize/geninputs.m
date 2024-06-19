function geninputs(params)

fs = 48000;
duration = 10;
outFolderPath = fullfile(params.RecordingsBaseDir, "whitenoise");

delete(fullfile(outFolderPath, '*'))
    
for i = 1:10
    y = wgn(1, duration * fs, 1);
    outFilenamePath = fullfile(outFolderPath, "whitenoise" + i + ".wav");
    mkdir(outFolderPath);
    audiowrite(outFilenamePath, y, fs);
end
